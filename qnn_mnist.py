
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from dorefa import *
# from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch QNN-MO-PYNQ MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=1,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--resume', default=False, action='store_true', help='Perform only evaluation on val dataset.')
parser.add_argument('--ab', type=int, default=2, metavar='N', help='number of bits for activations (default: 2)')
parser.add_argument('--eval', default=False, action='store_true', help='perform evaluation of trained model')
parser.add_argument('--export', default=False, action='store_true', help='perform weights export as npz of trained model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
prev_acc = 0
save_path = 'results/mnist-w1a{}.pt'.format(args.ab)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            BinarizeConv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=True),
            nn.MaxPool2d(2, stride=2, padding=0),
            Clamper(0, 1),
            Quantizer(args.ab),

            BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64,momentum=0.9,eps=1e-4),
            Clamper(0, 1),
            Quantizer(args.ab),
            nn.MaxPool2d(2, stride=2, padding=0),
           

            BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(64,momentum=0.9,eps=1e-4),
            Clamper(0, 1),
            Quantizer(args.ab))

        self.classifier = nn.Sequential(
            BinarizeLinear(64*5*5, 512, bias=True),
            nn.Linear(512, 10),
            nn.LogSoftmax())

    def forward(self, x):
        x = x.view(-1, 1,28,28)

        x = self.features(x)

        x = x.permute((0,2,3,1))
        x = x.contiguous()
        x = x.view(-1, 64*5*5)

        x = self.classifier(x)

        return x

    def export(self):
        import numpy as np
        dic = {}
        i = 0
        j = 0      
        # process conv and BN layers
        for k in range(len(self.features)):
            if hasattr(self.features[k], 'weight') and not hasattr(self.features[k], 'running_mean'):
                dic['conv'+str(i)+'/W:0'] = np.transpose(self.features[k].weight.detach().numpy(),(2,3,1,0))
                if self.features[k].bias is not None:
                    dic['conv'+str(i)+'/b:0'] = np.transpose(self.features[k].bias.detach().numpy())
                i = i + 1
            elif hasattr(self.features[k], 'running_mean'):
                dic['bn'+str(j)+'/beta:0'] = self.features[k].bias.detach().numpy()
                dic['bn'+str(j)+'/gamma:0'] = self.features[k].weight.detach().numpy()
                dic['bn'+str(j)+'/mean/EMA:0'] = self.features[k].running_mean.detach().numpy()
                dic['bn'+str(j)+'/variance/EMA:0'] = self.features[k].running_var.detach().numpy()
                j = j + 1
        i = 0
        j = 0
        # process linear and BN layers
        for k in range(len(self.classifier)):
            if hasattr(self.classifier[k], 'weight') and not hasattr(self.classifier[k], 'running_mean'):
                dic['fc'+str(i)+'/W:0'] = np.transpose(self.classifier[k].weight.detach().numpy())
                if self.classifier[k].bias is not None:
                    dic['fc'+str(i)+'/b:0'] = self.classifier[k].bias.detach().numpy()
                i = i + 1
            elif hasattr(self.classifier[k], 'running_mean'):
                dic['bn'+str(j)+'/beta:0'] = self.classifier[k].bias.detach().numpy()
                dic['bn'+str(j)+'/gamma:0'] = self.classifier[k].weight.detach().numpy()
                dic['bn'+str(j)+'/mean/EMA:0'] = self.classifier[k].running_mean.detach().numpy()
                dic['bn'+str(j)+'/variance/EMA:0'] = self.classifier[k].running_var.detach().numpy()
                j = j + 1
                        
        save_file = 'results/mnist-w1a{}.npz'.format(args.ab)
        np.savez(save_file, **dic)
        print("Model exported at: ", save_file)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):    
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()              
        for p in list(model.parameters()): 
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def test(save_model=False):
    model.eval()
    test_loss = 0
    correct = 0
    global prev_acc
    with torch.no_grad():
        for data, target in test_loader:  
            if args.cuda:
                data, target = data.cuda(), target.cuda() 
            output = model(data)
            test_loss += criterion(output, target).data # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    new_acc = 100. * correct.float() / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), new_acc))
    if new_acc > prev_acc:
        # save model
        if save_model:
            torch.save(model, save_path)
            print("Model saved at: ", save_path, "\n")
        prev_acc = new_acc

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net()
    if args.cuda:
        torch.cuda.set_device(0)    
        print(torch.cuda.get_device_name(0))
        model.cuda()
        dummy_input = Variable(torch.rand(1, 1, 28, 28)).cuda()
    else:
        dummy_input = Variable(torch.rand(1, 1, 28, 28))
    
    # with SummaryWriter(comment='Net1') as w:
    #     w.add_graph(model, (dummy_input, ), verbose=True)

    criterion = nn.CrossEntropyLoss()
    # test model
    if args.eval:
        model = torch.load(save_path)
        test()
    # export npz
    elif args.export:
        model = torch.load(save_path, map_location = 'cpu')
        model.export()
    # train model
    else:
        if args.resume:
            model = torch.load(save_path)
            test()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(save_model=True)
            if epoch%40==0:
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

