#coding: utf-8
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import time
import keys
import Levenshtein

#import models.crnn_lang as crnn
#print(crnn.__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--trainlist', required=True, help='path to train_list')
parser.add_argument('--vallist', required=True, help='path to val_list')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$')
parser.add_argument('--sep', type=str, default=':')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=10000, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--lang', action='store_true', help='whether to use char language model')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)
str1 = keys.alphabet
if opt.lang:
    import models.val_crnn_lang as crnn
else:
    import models.crnn as crnn
print(crnn.__name__)

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.listDataset(list_file =opt.trainlist)
assert train_dataset
if  opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
print (len(train_loader))
#test_dataset = dataset.listDataset(list_file =opt.vallist, transform=dataset.resizeNormalize((100, 32)))
test_dataset = dataset.listDataset(list_file =opt.vallist)

alphabet = str1.decode('utf-8')
nclass = len(alphabet)
nc = 3

converterAttention = utils.strLabelConverterForAttention(alphabet, opt.sep)
converterCTC = utils.strLabelConverterForCTC(alphabet, opt.sep)
criterionAttention = torch.nn.CrossEntropyLoss()
criterionCTC = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh, 24, 1, True)
#crnn.apply(weights_init)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn.load_state_dict(torch.load(opt.crnn))
print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
textAttention = torch.LongTensor(opt.batchSize * 5)
lengthAttention = torch.IntTensor(opt.batchSize)
textCTC = torch.IntTensor(opt.batchSize * 5)
lengthCTC = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    textAttention = textAttention.cuda()
    criterionAttention = criterionAttention.cuda()
    criterionCTC = criterionCTC.cuda()

image = Variable(image)
textAttention = Variable(textAttention)
lengthAttention = Variable(lengthAttention)
textCTC = Variable(textCTC)
lengthCTC = Variable(lengthCTC)


# loss averager
loss_avg = utils.averager()
loss_CTC = utils.averager()
loss_Attention = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, valdataset, criterionCTC, max_iter=100000):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    val_batchSize = 1
   # val_batchSize = opt.batchSize
    val_sampler = dataset.randomSequentialSampler(valdataset, val_batchSize)
    data_loader = torch.utils.data.DataLoader(
        valdataset, batch_size=val_batchSize,
        shuffle=False, sampler=val_sampler,
        num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
   # data_loader = torch.utils.data.DataLoader(
   #     dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    n_correctCTC = 0
    distanceCTC = 0
    sum_charNum = 0
    sum_imgNum = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        sum_imgNum += batch_size
     #   print (type(cpu_images),type(cpu_texts))
       # print (cpu_images.size(),max_iter,len(cpu_texts))
       # exit(0)
        utils.loadData(image, cpu_images)
        #print (image)

        predsCTC = crnn(image)
        preds_size = Variable(torch.IntTensor([predsCTC.size(0)] * batch_size))

                
        _, predsCTC = predsCTC.max(2)
        predsCTC = predsCTC.transpose(1, 0).contiguous().view(-1)
       # print (predsCTC)
        sim_predsCTC = converterCTC.decode(predsCTC.data, preds_size.data, raw=False)
       # print (sim_predsCTC)
        #exit(0)

        for i, cpu_text in enumerate(cpu_texts):
            gtText = cpu_text.decode('utf-8')
            #CTCText = sim_predsCTC[i]
            CTCText = sim_predsCTC
            if isinstance(CTCText,str):
                CTCText = CTCText.decode('utf-8')
            if gtText == CTCText:
                n_correctCTC += 1
            distanceCTCline = Levenshtein.distance(CTCText,gtText)
            #if distaceCTCline/len(gtText) > 0.2:
            #if distaceCTCline/len(gtText) > 0.2:
            if gtText != CTCText:
                print('gtText: %s' % gtText)
                print('CTCText: %s'% CTCText)
            distanceCTC += distanceCTCline
            sum_charNum = sum_charNum + len(gtText)
   
    print ('n_coorectCTC: %d, max_iter: %d, batch_size: %d'%(n_correctCTC,max_iter,batch_size))
    correctCTC_accuracy = n_correctCTC / float(sum_imgNum)
    cerCTC =  distanceCTC / float(sum_charNum)
    print('Test CERCTC: %f, accuracyCTC: %f' % (cerCTC, correctCTC_accuracy))


t0 = time.time()
val(crnn, test_dataset,  criterionCTC)
t1 = time.time()
print('time elapsed %d' % (t1-t0))
