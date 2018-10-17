import torch 
import numpy as np

 
net = torch.load('weights.pt', map_location = 'cpu')

dic = {


		'conv0/W:0':np.transpose(net['conv0.weight'],(2,3,1,0)), 
		'conv0/b:0':np.transpose(net['conv0.bias']), 
 

		'conv1/W:0':np.transpose(net['conv1.weight'],(2,3,1,0)), 


		'bn1/beta:0':net['bn1.bias'],
		'bn1/gamma:0':net['bn1.weight'],
		'bn1/mean/EMA:0':net['bn1.running_mean'],
		'bn1/variance/EMA:0':net['bn1.running_var'],


		'conv2/W:0': np.transpose(net['conv2.weight'],(2,3,1,0)), 


		'bn2/beta:0':net['bn2.bias'],
		'bn2/gamma:0':net['bn2.weight'],
		'bn2/mean/EMA:0':net['bn2.running_mean'],
		'bn2/variance/EMA:0':net['bn2.running_var'],



		'fc0/W:0':   np.transpose(net['fc0.weight']),
		'fc0/b:0':net['fc0.bias'], 
		'fc1/W:0':   np.transpose(net['fc1.weight']),
		'fc1/b:0':net['fc1.bias'], 
	}
  

np.savez('mnist.npz', **dic)
 
