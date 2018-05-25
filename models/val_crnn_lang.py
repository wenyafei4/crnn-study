import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time
from sru import SRU, SRUCell


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, paddding=0, dilation=1, groups=1):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, paddding, dilation, groups, False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    return model
	
def conv_bn_relu_CRNN(in_channels, out_channels, kernel_size, stride1=2, stride2=1, paddding=0, dilation=1, groups=1):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  (stride1,stride2), paddding, dilation, groups, False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    return model


def conv_bn(in_channels, out_channels, kernel_size, stride=1, paddding=0, dilation=1, groups=1):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, paddding, dilation, groups, False),
        nn.BatchNorm2d(out_channels),
    )
    return model


class MeanMaxPool2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super(MeanMaxPool2d, self).__init__()
        self.avg_pool = nn.AvgPool2d(*args, **kwargs)
        self.max_pool = nn.MaxPool2d(*args, **kwargs)

    def forward(self, x):
        return torch.cat((self.avg_pool(x), self.max_pool(x)), 1)


class SEBlock(nn.Module):

    def __init__(self, channels):
        super(SEBlock, self).__init__()
        assert channels % 16 == 0
        s = int(channels / 16)
        self.se = nn.Sequential(
            nn.Conv2d(channels, s, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(s, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f = nn.functional.avg_pool2d(x, (x.size(2), x.size(3)))
        f = self.se(f)
        x = x * f
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, bottleneck_channels, out_channels, stride, with_se):
        super(ResBlock, self).__init__()
        self.stride = stride

        self.use_pre_activation = False
        if self.use_pre_activation:
            self.conv = [
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                conv_bn_relu(in_channels, bottleneck_channels, 1),
                conv_bn_relu(bottleneck_channels, bottleneck_channels,
                             3, self.stride, 1, groups=2),
                nn.Conv2d(bottleneck_channels, out_channels, 1, bias=True),
            ]
        else:
            self.conv = [
                conv_bn_relu(in_channels, bottleneck_channels, 1),
                conv_bn_relu(bottleneck_channels, bottleneck_channels,
                             3, self.stride, 1, groups=2),
                conv_bn(bottleneck_channels, out_channels, 1),
            ]
            self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.conv.append(SEBlock(out_channels))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_pre_activation:
            if self.stride == 1:
                x = x + self.conv(x)
            else:
                x = self.conv(x)
            return x
        else:
            if self.stride == 1:
                x = self.relu(x + self.conv(x))
            else:
                x = self.relu(self.conv(x))
            return x

class ResBlockCRNN(nn.Module):

    def __init__(self, in_channels, bottleneck_channels, out_channels, stride1, stride2, with_se):
        super(ResBlockCRNN, self).__init__()
        self.stride1 = stride1
        self.stride2 = stride2

        self.use_pre_activation = False
        if self.use_pre_activation:
            self.conv = [
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                conv_bn_relu(in_channels, bottleneck_channels, 1),
                conv_bn_relu_CRNN(bottleneck_channels, bottleneck_channels,
                            3, self.stride1, self.stride2, 1, groups=2),
                nn.Conv2d(bottleneck_channels, out_channels, 1, bias=True),
            ]
        else:
            self.conv = [
                conv_bn_relu(in_channels, bottleneck_channels, 1),
				conv_bn_relu_CRNN(bottleneck_channels, bottleneck_channels,
                             3, self.stride1, self.stride2, 1, groups=2),
                conv_bn(bottleneck_channels, out_channels, 1),
            ]
            self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.conv.append(SEBlock(out_channels))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_pre_activation:
            if self.stride1 == 1:
                x = x + self.conv(x)
            else:
                x = self.conv(x)
            return x
        else:
            if self.stride1 == 1:
                x = self.relu(x + self.conv(x))
            else:
                x = self.relu(self.conv(x))
            return x

def res_stage(in_channels, out_channels, num_blocks, with_se):
    assert out_channels % 4 == 0
    bottleneck_channels = int(out_channels / 4)
    blocks = []
    for i in range(0, num_blocks):
        if i == 0:
            blocks.append(
                ResBlock(in_channels, bottleneck_channels, out_channels, 2, with_se))
        else:
            blocks.append(
                ResBlock(out_channels, bottleneck_channels, out_channels, 1, with_se))
    return nn.Sequential(*blocks)
	
def res_stageCRNN(in_channels, out_channels, num_blocks, with_se):
    assert out_channels % 4 == 0
    bottleneck_channels = int(out_channels / 4)
    blocks = []
    for i in range(0, num_blocks):
        if i == 0:
            blocks.append(
                ResBlockCRNN(in_channels, bottleneck_channels, out_channels, 2, 1, with_se))
        else:
            blocks.append(
                ResBlock(out_channels, bottleneck_channels, out_channels, 1, with_se))
    return nn.Sequential(*blocks)
		
class Network(nn.Module):

    def __init__(self, num_outputs, width, with_se=False, with_mean_max_pooling=False):
        super(Network, self).__init__()

        assert width % 2 == 0
        channel = lambda i: (2**i) * width
        self.network = nn.Sequential(
            nn.BatchNorm2d(3, affine=False),
            conv_bn_relu(3, 32, 3, 2, 1),
            nn.MaxPool2d(3, 2, 0, ceil_mode=True),
            res_stage(32, channel(2), 4, with_se),
            res_stage(channel(2), channel(3), 8, with_se),
            res_stage(channel(3), channel(4), 4, False),
        )
        if with_mean_max_pooling:
            self.pool = MeanMaxPool2d(7)
            self.fc = nn.Conv2d(channel(4) * 2, num_outputs, 1)
        else:
            self.pool = nn.AvgPool2d(7)
            self.fc = nn.Conv2d(channel(4), num_outputs, 1)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.running_mean, 0)
                nn.init.constant(m.running_var, 1)
                if m.weight is not None:
                    nn.init.constant(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
        nn.init.normal(self.fc.weight, std=0.001)

    def forward(self, x):
        x = self.network(x)
        x = self.pool(x)
        x = self.fc(x)
        return x.view(x.size(0), -1)

		

class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=128):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size+num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
	self.processed_batches = 0

    def forward(self, prev_hidden, feats, cur_embeddings):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        feats_proj = self.i2h(feats.view(-1,nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(F.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1)
        self.processed_batches = self.processed_batches + 1

        if self.processed_batches % 10000 == 0:
            print('processed_batches = %d' %(self.processed_batches))

        alpha = F.softmax(emition) # nB * nT
        if self.processed_batches % 10000 == 0:
            print('emition ', list(emition.data[0]))
            print('alpha ', list(alpha.data[0]))
        context = (feats * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0) # nB * nC
        context = torch.cat([context, cur_embeddings], 1)
        cur_hidden = self.rnn(context, prev_hidden)
        return cur_hidden, alpha

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_embeddings=128):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_embeddings)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes)
        self.char_embeddings = Parameter(torch.randn(num_classes+1, num_embeddings))
        self.num_embeddings = num_embeddings
        self.processed_batches = 0

    # targets is nT * nB
    def forward(self, feats, text_length, text):
        self.processed_batches = self.processed_batches + 1
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size
        assert(input_size == nC)
        assert(nB == text_length.numel())

        num_steps = text_length.data.max()
        num_labels = text_length.data.sum()
        targets = torch.zeros(nB, num_steps+1).long().cuda()
        start_id = 0
        for i in range(nB):
            targets[i][1:1+text_length.data[i]] = text.data[start_id:start_id+text_length.data[i]]+1
            start_id = start_id+text_length.data[i]
        targets = Variable(targets.transpose(0,1).contiguous())

        output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
        hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))
        max_locs = torch.zeros(num_steps, nB)
        max_vals = torch.zeros(num_steps, nB)
        for i in range(num_steps):
            cur_embeddings = self.char_embeddings.index_select(0, targets[i])
            hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings)
            output_hiddens[i] = hidden
            if self.processed_batches % 500 == 0:
                max_val, max_loc = alpha.data.max(1)
                max_locs[i] = max_loc.cpu()
                max_vals[i] = max_val.cpu()
        if self.processed_batches % 500 == 0:
            print('max_locs', list(max_locs[0:text_length.data[0],0]))
            print('max_vals', list(max_vals[0:text_length.data[0],0]))
        new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))
        b = 0
        start = 0
        for length in text_length.data:
            new_hiddens[start:start+length] = output_hiddens[0:length,b,:]
            start = start + length
            b = b + 1
        probs = self.generator(new_hiddens)
        return probs

class BidirectionalLSTM_Embed(nn.Module):

    def __init__(self, nIn, nHidden, nOut, isSRU, nLayer):
        super(BidirectionalLSTM_Embed, self).__init__()
        if False == isSRU:
            self.rnn = nn.LSTM(nIn, nHidden, nLayer, bidirectional=True)
        else:
            self.rnn = SRU(nIn, nHidden,
                num_layers = nLayer,          # number of stacking RNN layers
                dropout = 0.0,           # dropout applied between RNN layers
                rnn_dropout = 0.0,       # variational dropout applied on linear transformation
                use_tanh = 1,            # use tanh?
                use_relu = 0,            # use ReLU?
                bidirectional = True    # bidirectional RNN ?
            )
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, isSRU, nLayer):
        super(BidirectionalLSTM_Embed, self).__init__()
        if False == isSRU:
            self.rnn = nn.LSTM(nIn, nHidden, nLayer, bidirectional=True)
        else:
            self.rnn = SRU(nIn, nHidden,
                num_layers = nLayer,          # number of stacking RNN layers
                dropout = 0.0,           # dropout applied between RNN layers
                rnn_dropout = 0.0,       # variational dropout applied on linear transformation
                use_tanh = 1,            # use tanh?
                use_relu = 0,            # use ReLU?
                bidirectional = True    # bidirectional RNN ?
            )

    def forward(self, input):
        output, _ = self.rnn(input)
        return output

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, width=48, n_rnn=1, isSRU=True, leakyRelu=False,  with_se=False, with_mean_max_pooling=False):
        super(CRNN, self).__init__()
        assert width % 2 == 0
        channel = lambda i: (2**i) * width
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(3, affine=False),
            conv_bn_relu(3, 32, 3, 2, 1),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),
            res_stageCRNN(32, channel(2), 4, with_se),
            res_stageCRNN(channel(2), channel(3), 8, with_se),
			#res_stageCRNN(channel(3), channel(4), 4, False),
			conv_bn_relu(channel(3), channel(4), 2, 1, 0),
        )

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.running_mean, 0)
                nn.init.constant(m.running_var, 1)
                if m.weight is not None:
                    nn.init.constant(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
        if False == isSRU:
            self.rnn = nn.LSTM(channel(4), nh, n_rnn, bidirectional=True)
        else:
            self.rnn = SRU(channel(4), nh,
                num_layers = n_rnn,          # number of stacking RNN layers
                dropout = 0.0,           # dropout applied between RNN layers
                rnn_dropout = 0.0,       # variational dropout applied on linear transformation
                use_tanh = 1,            # use tanh?
                use_relu = 0,            # use ReLU?
                bidirectional = True    # bidirectional RNN ?
            )
        self.embeddingCTC = nn.Linear(nh * 2, nclass)
        self.attention = Attention(nh * 2, nh, nclass, 256)

    #def forward(self, input, length, text):
    def forward(self, input):
        # conv features
        timeBegin = time.time()
        conv = self.cnn(input)
       # print ("cnn time is %f" % (time.time()-timeBegin))
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        rnn,_ = self.rnn(conv)
        T, b, h = rnn.size()
        t_rec = rnn.view(T * b, h)
        CTCEmbed = self.embeddingCTC(t_rec)  # [T * b, nOut]
        outputCTC = CTCEmbed.view(T, b, -1)
     #   outputAttention = self.attention(rnn, length, text)

        #return outputCTC, outputAttention
        return outputCTC
	

class CRNN_ori(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = nn.Sequential(
                      nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 64x16x50
                      nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 128x8x25
                      nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 256x8x25
                      nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 256x4x25
                      nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), # 512x4x25
                      nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 512x2x25
                      nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)) # 512x1x25
        #self.cnn = cnn
        self.rnn = nn.LSTM(nh, nh, bidirectional=True)
        self.rnnShare = nn.Sequential(BidirectionalLSTM_Embed(512, nh, nh),self.rnn)
        self.embeddingCTC = nn.Linear(nh * 2, nclass)
        self.embeddingAttention = nn.Linear(nh * 2, nh)
        self.attention = Attention(nh, nh, nclass, 256)

    def forward(self, input, length, text):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        crnnShare,_ = self.rnnShare(conv)
        T, b, h = crnnShare.size()
        t_rec = crnnShare.view(T * b, h)
        CTCEmbed = self.embeddingCTC(t_rec)  # [T * b, nOut]
        outputCTC = CTCEmbed.view(T, b, -1)
        AttentionEmbed = self.embeddingAttention(t_rec)  # [T * b, nOut]
        AttentionInput = AttentionEmbed.view(T, b, -1)
        outputAttention = self.attention(AttentionInput, length, text)

        return outputCTC, outputAttention
