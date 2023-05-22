import pandas as pd
import argparse, getpass, datetime, shutil
import torch.backends.cudnn as cudnn
import joblib
import time

from glob import glob
from data import *
from model import*


#torch.cuda.set_device(0)
print(torch.cuda.current_device())


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch RNN segmentation model')
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default='2015_BOE_Chiu',
                        help='dataset name')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='sequential_LSTM',
                        help='model architecture: '
                             'SimpleRNN'
                             'SegNet'
                             'ReNetPure'
                             'LSTM'
                             ' (default: SimpleRNN)')
    parser.add_argument('--seed', type=int, default=5, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--split', type=int, default=7, metavar='S',
                        help='split the image evenly into n columns')
    parser.add_argument('--nworkers', type=int, default=0,
                        help='number of data loading workers [0 to do it using main process]')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--crop', default=(0, 0), type=tuple,
                        help='(crop factor top , crop factor bottom')
    parser.add_argument('--early-stop', default=30, type=int,
                        metavar='N', help='early stopping (default: 20)')


    #model args
    parser.add_argument('--loss', default='CrossEntropyLoss',
                        help='loss: ' +
                             ' (default: CrossEntropyLoss), BCEWithLogitsLoss, MSELoss, NLLLoss')
    parser.add_argument('--weight-decay', default=1e-5, type=float,
                        help='Weight decay rate set to 0 to disable')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_hidden', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr-scheduler', default=True, type=str2bool,
                        help='enable to use a lr scheduler')
    parser.add_argument('--lr_drop_factor', default=0.0001, type=float,
                        help='enable to use a lr scheduler')
    parser.add_argument('--lr_drop_patience', default=20, type=int,
                        help='number of steps wihtout any metric improvement')
    parser.add_argument('--Clip_Grad_Norm', default=0.0, type=float,
                        help = 'max l2 norm of gradient of parameters - use 0 to disable it')
    parser.add_argument('--nesterov', default=True, type=str2bool,
                        help='nesterov')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')

    args = parser.parse_args()

    return args

args = parse_args()
best_iou = 0

def generate_run_id():


    now = datetime.datetime.now()
    date = map(str, [now.year, now.month, now.day])
    coarse_time = map(str, [now.hour, now.minute])

    run_id = '_'.join(['-'.join(date), '-'.join(coarse_time)])
    return run_id

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    RUN_ID = generate_run_id()

    if args.name is None:
        args.name = '%s_%s_woDS' %(args.dataset, args.arch)
    if not os.path.exists('models\\%s_%s' %(args.name, RUN_ID)):
        os.makedirs('models\\%s_%s' %(args.name, RUN_ID))

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')


    with open('models\\%s_%s\\args.txt' %(args.name, RUN_ID), 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models\\%s_%s\\args.pkl' %(args.name, RUN_ID))

    cudnn.benchmark = True

    # Data loading, loads images right into memory at the moment
    img_paths = glob('input\\' + args.dataset + '_images\\*')
    mask_paths = glob('input\\' + args.dataset + '_masks1\\*')

    receptive_filter_size_w = round(547 / args.split)+ 1

    # load in data and split it
    train_img, val_img, train_mask, val_mask = \
    train_test_split(img_paths, mask_paths, dataset=args.dataset, split=args.split, test_size=0.4, random_state=41, max_size = receptive_filter_size_w )

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

    # get model and params
    model, criterion, optimizer, train_loader, val_loader = getModel(train_img, train_mask,  val_img, val_mask, args=args)

    print("=> creating model %s" % args.arch)

    best_iou = 0

    # count indicator to stop if no change during training
    trigger = 0

    for epoch in range(args.epochs):
        print('Epoch [%d/%d] \n' % (epoch, args.epochs))
        epoch_start = time.time()
        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start

        print(str(epoch) + '_duration: ' + str(epoch_duration) + '_loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models\\%s_%s\\log.csv' % (args.name, RUN_ID), index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models\\%s_%s\\model.pth' % (args.name, RUN_ID))
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
