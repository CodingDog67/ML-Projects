import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from torch.nn import functional as F
from torch.autograd import Variable

# simple LSTM
class sequential_LSTM(nn.Module):
    def __init__(self, sequence_len, sentence_len, hidden_size, n_layers=1,
                               drop_prob=0.2, lr=0.001):
        super(sequential_LSTM, self).__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = hidden_size
        self.lr = lr

        ##  define the LSTM
        self.lstm = nn.LSTM(sentence_len, hidden_size, n_layers, bidirectional=False)

        ##  define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ##  define the final, fully-connected output layer, 18 = num classes
        self.fc = nn.Linear(hidden_size, sentence_len )

        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x, hidden):


        ##  Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)

        ## pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        # use contiguous to reshape the output
        #out = out.contiguous().view(-1, self.n_hidden)

        ## put x through the fully-connected layer
        out = self.fc(out)

        out = self.softmax(out)
        return out, hidden

    def initHidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

# simple character level RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size)).cuda()

# encoder - decoder seq to seq
class Seq2Seq:
    kiki = 0


# combines SegNet and ReNet
class SegNet(nn.Module):

    def __init__(self, args, n_input, n_units, patch_size=(1, 1), usegpu=True):
        super(SegNet, self).__init__()

        self.args = args

        self.patch_size_height = int(patch_size[0])
        self.patch_size_width = int(patch_size[1])

        assert self.patch_size_height >= 1
        assert self.patch_size_width >= 1

        self.tiling = False if ((self.patch_size_height == 1) and (self.patch_size_width == 1)) else True

        self.rnn_hor = nn.GRU(n_input * self.patch_size_height * self.patch_size_width, n_units,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.rnn_ver = nn.GRU(n_units * 2, n_units, num_layers=1, batch_first=True, bidirectional=True)

    def tile(self, x):

        n_height_padding = self.patch_size_height - x.size(2) % self.patch_size_height
        n_width_padding = self.patch_size_width - x.size(3) % self.patch_size_width

        n_top_padding = n_height_padding / 2
        n_bottom_padding = n_height_padding - n_top_padding

        n_left_padding = n_width_padding / 2
        n_right_padding = n_width_padding - n_left_padding

        x = F.pad(x, (n_left_padding, n_right_padding, n_top_padding, n_bottom_padding))

        b, n_filters, n_height, n_width = x.size()

        assert n_height % self.patch_size_height == 0
        assert n_width % self.patch_size_width == 0

        new_height = n_height / self.patch_size_height
        new_width = n_width / self.patch_size_width

        x = x.view(b, n_filters, new_height, self.patch_size_height, new_width, self.patch_size_width)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous()
        x = x.view(b, new_height, new_width, self.patch_size_height * self.patch_size_width * n_filters)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

        return x

    def rnn_forward(self, x, hor_or_ver):

        assert hor_or_ver in ['hor', 'ver']

        b, n_height, n_width, n_filters = x.size()

        x = x.view(b * n_height, n_width, n_filters)
        if hor_or_ver == 'hor':
            x, _ = self.rnn_hor(x)
        else:
            x, _ = self.rnn_ver(x)
        x = x.contiguous()
        x = x.view(b, n_height, n_width, -1)

        return x

    def forward(self, x):

                                       #b, nf, h, w
        if self.tiling:
            x = self.tile(x)           #b, nf, h, w
        x = x.permute(0, 2, 3, 1)      #b, h, w, nf
        x = x.contiguous()
        x = self.rnn_forward(x, 'hor') #b, h, w, nf
        x = x.permute(0, 2, 1, 3)      #b, w, h, nf
        x = x.contiguous()
        x = self.rnn_forward(x, 'ver') #b, w, h, nf
        x = x.permute(0, 2, 1, 3)      #b, h, w, nf
        x = x.contiguous()
        x = x.permute(0, 3, 1, 2)      #b, nf, h, w
        x = x.contiguous()

        return x
class ReNet(nn.Module):
    # renet with one layer
    class ReNet(nn.Module):
        def __init__(self, args, receptive_filter_size, hidden_size, batch_size, image_patches_height,
                     image_patches_width):

            super(ReNet, self).__init__()
            self.args = args

            self.batch_size = batch_size
            self.receptive_filter_size = receptive_filter_size
            self.input_size1 = receptive_filter_size * receptive_filter_size * 3
            self.input_size2 = hidden_size * 2
            self.hidden_size = hidden_size

            # vertical rnns
            self.rnn1 = nn.LSTM(self.input_size1, self.hidden_size, dropout=0.2)
            self.rnn2 = nn.LSTM(self.input_size1, self.hidden_size, dropout=0.2)

            # horizontal rnns
            self.rnn3 = nn.LSTM(self.input_size2, self.hidden_size, dropout=0.2)
            self.rnn4 = nn.LSTM(self.input_size2, self.hidden_size, dropout=0.2)

            self.initHidden()

            # feature_map_dim = int(image_patches_height*image_patches_height*hidden_size*2)
            self.conv1 = nn.Conv2d(hidden_size * 2, 2, 3, padding=1)  # [1,640,8,8]->[1,1,8,8]
            self.UpsamplingBilinear2d = nn.UpsamplingBilinear2d(size=(32, 32), scale_factor=None)
            # self.dense = nn.Linear(feature_map_dim, 4096)
            # self.fc = nn.Linear(4096, 10)

            self.log_softmax = nn.LogSoftmax()

        def initHidden(self):
            self.hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_size)),
                           Variable(torch.zeros(1, self.batch_size, self.hidden_size)))

        def get_image_patches(self, X, receptive_filter_size):
            """
            creates image patches based on the dimension of a receptive filter
            """
            image_patches = []

            _, X_channel, X_height, X_width = X.size()

            for i in range(0, X_height, receptive_filter_size):
                for j in range(0, X_width, receptive_filter_size):
                    X_patch = X[:, :, i: i + receptive_filter_size, j: j + receptive_filter_size]
                    image_patches.append(X_patch)

            image_patches_height = (X_height // receptive_filter_size)
            image_patches_width = (X_width // receptive_filter_size)

            image_patches = torch.stack(image_patches)
            image_patches = image_patches.permute(1, 0, 2, 3, 4)

            image_patches = image_patches.contiguous().view(-1, image_patches_height, image_patches_height,
                                                            receptive_filter_size * receptive_filter_size * X_channel)

            return image_patches

        def get_vertical_rnn_inputs(self, image_patches, forward):
            """
            creates vertical rnn inputs in dimensions
            (num_patches, batch_size, rnn_input_feature_dim)
            num_patches: image_patches_height * image_patches_width
            """
            vertical_rnn_inputs = []
            _, image_patches_height, image_patches_width, feature_dim = image_patches.size()

            if forward:
                for i in range(image_patches_height):
                    for j in range(image_patches_width):
                        vertical_rnn_inputs.append(image_patches[:, j, i, :])

            else:
                for i in range(image_patches_height - 1, -1, -1):
                    for j in range(image_patches_width - 1, -1, -1):
                        vertical_rnn_inputs.append(image_patches[:, j, i, :])

            vertical_rnn_inputs = torch.stack(vertical_rnn_inputs)

            return vertical_rnn_inputs

        def get_horizontal_rnn_inputs(self, vertical_feature_map, image_patches_height, image_patches_width, forward):
            """
            creates vertical rnn inputs in dimensions
            (num_patches, batch_size, rnn_input_feature_dim)
            num_patches: image_patches_height * image_patches_width
            """
            horizontal_rnn_inputs = []

            if forward:
                for i in range(image_patches_height):
                    for j in range(image_patches_width):
                        horizontal_rnn_inputs.append(vertical_feature_map[:, i, j, :])
            else:
                for i in range(image_patches_height - 1, -1, -1):
                    for j in range(image_patches_width - 1, -1, -1):
                        horizontal_rnn_inputs.append(vertical_feature_map[:, i, j, :])

            horizontal_rnn_inputs = torch.stack(horizontal_rnn_inputs)

            return horizontal_rnn_inputs

        def forward(self, X):

            """ReNet """

            # divide input input image to image patches
            image_patches = self.get_image_patches(X, self.receptive_filter_size)
            _, image_patches_height, image_patches_width, feature_dim = image_patches.size()

            # process vertical rnn inputs
            vertical_rnn_inputs_fw = self.get_vertical_rnn_inputs(image_patches, forward=True)
            vertical_rnn_inputs_rev = self.get_vertical_rnn_inputs(image_patches, forward=False)

            # extract vertical hidden states
            vertical_forward_hidden, vertical_forward_cell = self.rnn1(vertical_rnn_inputs_fw, self.hidden)
            vertical_reverse_hidden, vertical_reverse_cell = self.rnn2(vertical_rnn_inputs_rev, self.hidden)

            # create vertical feature map
            vertical_feature_map = torch.cat((vertical_forward_hidden, vertical_reverse_hidden), 2)
            vertical_feature_map = vertical_feature_map.permute(1, 0, 2)

            # reshape vertical feature map to (batch size, image_patches_height, image_patches_width, hidden_size * 2)
            vertical_feature_map = vertical_feature_map.contiguous().view(-1, image_patches_width, image_patches_height,
                                                                          self.hidden_size * 2)
            vertical_feature_map.permute(0, 2, 1, 3)

            # process horizontal rnn inputs
            horizontal_rnn_inputs_fw = self.get_horizontal_rnn_inputs(vertical_feature_map, image_patches_height,
                                                                      image_patches_width, forward=True)
            horizontal_rnn_inputs_rev = self.get_horizontal_rnn_inputs(vertical_feature_map, image_patches_height,
                                                                       image_patches_width, forward=False)

            # extract horizontal hidden states
            horizontal_forward_hidden, horizontal_forward_cell = self.rnn3(horizontal_rnn_inputs_fw, self.hidden)
            horizontal_reverse_hidden, horizontal_reverse_cell = self.rnn4(horizontal_rnn_inputs_rev, self.hidden)

            # create horiztonal feature map[64,1,320]
            horizontal_feature_map = torch.cat((horizontal_forward_hidden, horizontal_reverse_hidden), 2)
            horizontal_feature_map = horizontal_feature_map.permute(1, 0, 2)

            # flatten[1,64,640]
            output = horizontal_feature_map.contiguous().view(-1, image_patches_height, image_patches_width,
                                                              self.hidden_size * 2)
            output = output.permute(0, 3, 1, 2)  # [1,640,8,8]
            conv1 = self.conv1(output)
            Upsampling = self.UpsamplingBilinear2d(conv1)
            # dense layer
            # output = F.relu(self.dense(output))

            # fully connected layer
            # logits = self.fc(output)

            # log softmax
            logits = self.log_softmax(Upsampling)

            return logits

# renet with one layer
class ReNetPure(nn.Module):
    def __init__(self, args, receptive_filter_size_h, receptive_filter_size_w, hidden_size):

        super(ReNetPure, self).__init__()
        self.args = args


        self.batch_size = args.batch_size
        self.receptive_filter_size_h = receptive_filter_size_h
        self.receptive_filter_size_w =receptive_filter_size_w


        self.input_size1 = receptive_filter_size_h * receptive_filter_size_w * 1
        self.input_size2 = hidden_size * 2
        self.hidden_size = hidden_size

        # vertical rnns
        self.rnn1 = nn.LSTM(self.input_size1, self.hidden_size, dropout=0.1)
        self.rnn2 = nn.LSTM(self.input_size1, self.hidden_size, dropout=0.1)

        # horizontal rnns
        self.rnn3 = nn.LSTM(self.input_size2, self.hidden_size, dropout=0.1)
        self.rnn4 = nn.LSTM(self.input_size2, self.hidden_size)

        self.initHidden()

        # feature_map_dim = int(image_patches_height*image_patches_height*hidden_size*2)
        self.conv1 = nn.Conv2d(hidden_size * 2, 1, 1, padding=0)  # [1,640,8,8]->[1,1,8,8]
        self.UpsamplingBilinear2d = nn.UpsamplingBilinear2d(size=(receptive_filter_size_h, receptive_filter_size_w), scale_factor=None)
        # self.dense = nn.Linear(feature_map_dim, 4096)
        # self.fc = nn.Linear(4096, 10)

        self.log_softmax = nn.LogSoftmax()

    def initHidden(self):
        self.hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda(),
                       Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda())


    def get_image_patches(self, X, receptive_filter_size_h, receptive_filter_size_w):
        """
        creates image patches based on the dimension of a receptive filter
        """
        image_patches = []
        receptive_filter_size_height = receptive_filter_size_h
        receptive_filter_size_width = receptive_filter_size_w
        _, X_channel, X_height, X_width = X.size()


        for i in range(0, X_height, receptive_filter_size_height):
            for j in range(0, X_width, receptive_filter_size_width):
                X_patch = X[:, :, i: i + receptive_filter_size_height, j: j + receptive_filter_size_width]
                image_patches.append(X_patch)

        image_patches_height = (X_height // receptive_filter_size_height)
        image_patches_width = (X_width // receptive_filter_size_width)

        image_patches = torch.stack(image_patches)
        image_patches = image_patches.permute(1, 0, 2, 3, 4)

        #image_patches = image_patches.squeeze(1)
        image_patches = image_patches.contiguous().view(-1, image_patches_height, image_patches_width,
         receptive_filter_size_h * receptive_filter_size_w * X_channel)

        return image_patches

    def get_vertical_rnn_inputs(self, image_patches, forward):
        """
        creates vertical rnn inputs in dimensions
        (num_patches, batch_size, rnn_input_feature_dim)
        num_patches: image_patches_height * image_patches_width
        """
        vertical_rnn_inputs = []
        _, image_patches_width, image_patches_height, feature_dim = image_patches.size()

        if forward:
            for i in range(image_patches_height):
                for j in range(image_patches_width):
                    vertical_rnn_inputs.append(image_patches[:, j, i, :])

        else:
            for i in range(image_patches_height - 1, -1, -1):
                for j in range(image_patches_width - 1, -1, -1):
                    vertical_rnn_inputs.append(image_patches[:, j, i, :])

        vertical_rnn_inputs = torch.stack(vertical_rnn_inputs)

        return vertical_rnn_inputs

    def get_horizontal_rnn_inputs(self, vertical_feature_map, image_patches_height, image_patches_width, forward):
        """
        creates vertical rnn inputs in dimensions
        (num_patches, batch_size, rnn_input_feature_dim)
        num_patches: image_patches_height * image_patches_width
        """
        horizontal_rnn_inputs = []

        if forward:
            for i in range(image_patches_height):
                for j in range(image_patches_width):
                    horizontal_rnn_inputs.append(vertical_feature_map[:, i, j, :])
        else:
            for i in range(image_patches_height - 1, -1, -1):
                for j in range(image_patches_width - 1, -1, -1):
                    horizontal_rnn_inputs.append(vertical_feature_map[:, i, j, :])

        horizontal_rnn_inputs = torch.stack(horizontal_rnn_inputs)

        return horizontal_rnn_inputs

    def forward(self, X):

        """ReNet """


        # divide input input image to image patches
        image_patches = self.get_image_patches(X, self.receptive_filter_size_h, self.receptive_filter_size_w)
        _, image_patches_height, image_patches_width, feature_dim = image_patches.size()

        # process vertical rnn inputs
        vertical_rnn_inputs_fw = self.get_vertical_rnn_inputs(image_patches, forward=True)
        vertical_rnn_inputs_rev = self.get_vertical_rnn_inputs(image_patches, forward=False)

        # extract vertical hidden states
        vertical_forward_hidden, vertical_forward_cell = self.rnn1(vertical_rnn_inputs_fw, self.hidden)
        vertical_reverse_hidden, vertical_reverse_cell = self.rnn2(vertical_rnn_inputs_rev, self.hidden)

        # create vertical feature map
        vertical_feature_map = torch.cat((vertical_forward_hidden, vertical_reverse_hidden), 2)
        vertical_feature_map = vertical_feature_map.permute(1, 0, 2)

        # reshape vertical feature map to (batch size, image_patches_height, image_patches_width, hidden_size * 2)
        vertical_feature_map = vertical_feature_map.contiguous().view(-1, image_patches_width, image_patches_height,
                                                                      self.hidden_size * 2)
        vertical_feature_map.permute(0, 2, 1, 3)

        # process horizontal rnn inputs
        horizontal_rnn_inputs_fw = self.get_horizontal_rnn_inputs(vertical_feature_map, image_patches_height,
                                                                  image_patches_width, forward=True)
        horizontal_rnn_inputs_rev = self.get_horizontal_rnn_inputs(vertical_feature_map, image_patches_height,
                                                                   image_patches_width, forward=False)

        # extract horizontal hidden states
        horizontal_forward_hidden, horizontal_forward_cell = self.rnn3(horizontal_rnn_inputs_fw, self.hidden)
        horizontal_reverse_hidden, horizontal_reverse_cell = self.rnn4(horizontal_rnn_inputs_rev, self.hidden)

        # create horiztonal feature map[64,1,320]
        horizontal_feature_map = torch.cat((horizontal_forward_hidden, horizontal_reverse_hidden), 2)
        horizontal_feature_map = horizontal_feature_map.permute(1, 0, 2)

        # flatten[1,64,640]
        output = horizontal_feature_map.contiguous().view(-1, image_patches_height, image_patches_width,
                                                          self.hidden_size * 2)
        output = output.permute(0, 3, 1, 2)  # [1,640,8,8]
        conv1 = self.conv1(output)
        Upsampling = self.UpsamplingBilinear2d(conv1)
        # dense layer
        # output = F.relu(self.dense(output))

        # fully connected layer
        # logits = self.fc(output)

        # log softmax
        logits = self.log_softmax(Upsampling)

        return logits



