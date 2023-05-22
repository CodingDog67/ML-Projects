from data import Dataset
import torch
import torch.nn as nn
import archs
import torch.optim as optim

from torch.autograd import Variable, gradcheck
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
from metrics import iou_score, str2bool

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer, scheduler=None ):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.type(torch.LongTensor).cuda()

        input, target = Variable(input), Variable(target)
        hidden = model.initHidden(1)

        # input of shape (seq_len, batch, input_size
        input = input.squeeze(0)

        input = input.view(input.shape[1], input.shape[0], input.shape[2])
        # compute output

        #for i in range(input.shape[0]):

        output, hidden = model(input.clone(), hidden)

        # In the 3D case, the torch.nn.CrossEntropy() functions expects two arguments: a 4D input matrix and a 3D target matrix.
        # The input matrix is in the shape: (Minibatch, Classes, H, W). The target matrix is in the shape (Minibatch, H, W)
        # with numbers ranging from 0 to (Classes-1)
        loss = criterion(output.unsqueeze(0), target[:, :, :, :].squeeze(1))

        # expected prediction shape  batch x classes (seq len, batch, sentence length, classes )
        # expected target shape batch size
        losses.update(loss.item(), input.size(0))


        # compute gradient and do optimizing step
        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
    ])

    return log

def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.type(torch.LongTensor).cuda()

            # compute output
            output = model(input)
            loss =  criterion(output, target.squeeze(1))
            iou = iou_score(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log

def getModel(*arrays, args):


    train_img = arrays[0]
    train_mask = arrays[1]
    val_img = arrays[2]
    val_mask = arrays[3]

    # create model own class maybe?
    train_dataset = Dataset(args, train_img, train_mask, args.crop, args.aug)
    assert train_dataset
    val_dataset = Dataset(args, val_img, val_mask, args.crop)
    assert val_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nworkers,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.nworkers,
        pin_memory=True,
        drop_last=False)

    hidden_size = args.num_hidden
    sequence_len = train_img[0].shape[0]
    sentence_len = train_img[0].shape[1]

    if args.arch == 'ReNetPure':
        receptive_filter_size_h = sequence_len
        receptive_filter_size_w = sentence_len
        model = archs.__dict__[args.arch](args, receptive_filter_size_h, receptive_filter_size_w, hidden_size)

    if args.arch == "SimpleRNN":
        model = archs.__dict__[args.arch](args, sentence_len, hidden_size, sequence_len)

    if args.arch == "RNN":
        model = archs.__dict__[args.arch]( sentence_len, hidden_size, sequence_len)

    if args.arch == "sequential_LSTM":
        model = archs.__dict__[args.arch](sequence_len, sentence_len, hidden_size )

    model = model.cuda()

    # Define loss function
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    if args.loss == 'MSELoss':
        criterion = nn.MSELoss().cuda()
    if args.loss == 'NLLLoss':
        criterion = nn.NLLLoss()


    # Define optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  weight_decay=args.weight_decay)

    elif args.optimizer == 'Adadelta':
        optimizer = optim.adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                   weight_decay=args.weight_decay)

    if args.lr_scheduler:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_drop_factor,
                                                            patience=args.lr_drop_patience,
                                                            verbose=True)

    return model, criterion, optimizer, train_loader, val_loader