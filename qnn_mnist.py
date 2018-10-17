
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from dorefa import *
from tensorboardX import SummaryWriter


save_path = 'weights.pt'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=1,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.conv0 = BinarizeConv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=True)
        self.mp0 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv1 = BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64,momentum=0.9,eps=1e-4)    #,track_running_stats=False
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv2 = BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(64,momentum=0.9,eps=1e-4)

        self.fc0 = BinarizeLinear(64*5*5,512)
        self.fc1 = nn.Linear(512, 10)
        self.logsoftmax=nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, 1,28,28)
 
 
        x = self.conv0(x) 
        x = self.mp0(x)
        x = x.clamp_(0,1)
	x = Quantizer.apply(x) 

        x = self.conv1(x)
        x = self.bn1(x)
        x = x.clamp_(0,1)
	x = Quantizer.apply(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x.clamp_(0,1)
	x = Quantizer.apply(x)


        x = x.permute((0,2,3,1))
        x = x.contiguous()
        x = x.view(-1, 64*5*5)

        x = self.fc0(x)
        x = x.clamp_(0,1)
        x = self.fc1(x)
        return self.logsoftmax(x)

model = Net()

#model.fc1.register_backward_hook(printgradnorm)


#model.load_state_dict(torch.load(save_path))

if args.cuda:
    torch.cuda.set_device(0)    
    print(torch.cuda.get_device_name(0))
    model.cuda()


dummy_input = Variable(torch.rand(1, 1, 28, 28)).cuda()
with SummaryWriter(comment='Net1') as w:
	w.add_graph(model, (dummy_input, ), verbose=True)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)



def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #print('max',np.ndarray.min(data.numpy())) 
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

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
                100. * batch_idx / len(train_loader), loss.data[0]))

        
    torch.save(model.state_dict(), save_path)     # after each epoch  

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:  
        if args.cuda:
            data, target = data.cuda(), target.cuda() 
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()

