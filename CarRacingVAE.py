import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Normal 
from torch.optim.lr_scheduler import StepLR

import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
plt.style.use('ggplot')

# --------------------------------------------------------
# --------------------------------------------------------

# 	Racing Car autoencoder. Takes a dataset of 64x64x3 images
#  	through 4 convolutional layers, samples the code and takes it again through 4 convTranspose layers 
# 	For now, unable to generate nice images 


# --------------------------------------------------------
# --------------------------------------------------------


class VAE(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		self.c1 = nn.Conv2d(3,16,8, stride = 2)
		self.c2 = nn.Conv2d(16,32,4, stride = 2)
		self.c3 = nn.Conv2d(32,64,4, stride = 2)
		self.c4 = nn.Conv2d(64,128,4, stride = 2)

		self.means = nn.Linear(128,32)
		self.stds = nn.Linear(128,32)

		self.d1 = nn.ConvTranspose2d(32,128,6,stride = 2)
		self.d2 = nn.ConvTranspose2d(128,64,5,stride = 2)
		self.d3 = nn.ConvTranspose2d(64,32,4,stride = 2)
		self.d4 = nn.ConvTranspose2d(32,3,2,stride = 2)

		self.adam = optim.Adam(self.parameters(), 3e-4)

	def forward(self,x): 

		means, logvar = self.encode(x)
		z = self.reparam(means, logvar)
		pred = self.decode(z)

		return pred,means,logvar

	def reparam(self, mu, logvar): 

		std = torch.exp(logvar*0.5)
		eps = torch.rand_like(std)

		z = mu + eps*std
		return z

	def encode(self, x): 

		batch = x.shape[0]
		x = F.elu(self.c1(x))
		x = F.elu(self.c2(x))
		x = F.elu(self.c3(x))
		x = F.elu(self.c4(x))

		x = x.reshape(batch, -1)
		mean, std = self.means(x), F.elu(self.stds(x))
		return mean, std

	def decode(self,x): 

		batch = x.shape[0]
		x = x.reshape(batch, -1, 1,1)
		x = F.elu(self.d1(x))
		x = F.elu(self.d2(x))
		x = F.elu(self.d3(x))
		x = F.sigmoid(self.d4(x))

		return x

	def demo(self): 

		mu = torch.zeros(1,32)
		stds = torch.ones_like(mu)
		dist = Normal(mu, stds)

		z = dist.sample()

		return self.decode(z)

	def maj(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		self.adam.step()

	def eval_loss(self, pred, target, mean, logvar): 

		recon_loss = F.binary_cross_entropy(pred, target)
		kl_loss = 0.5*torch.sum(torch.exp(logvar) + mu.pow(2) - 1. - logvar) #(logvar.exp() + mean.pow(2) - (logvar + torch.ones_like(logvar))).sum()

		loss = kl_loss + recon_loss
		return loss 

class Loader():

	def __init__(self): 

		self.path = 'Data64/'
		self.max_count = 600
		self.counter = 0 

	def sample(self,batch_size = 32, use_cuda = False): 

		data = np.zeros((batch_size, 3,64,64))
		for i in range(batch_size): 
			name = self.path + str(self.counter)
			image = pickle.load(open(name,'rb'))
			data[i,:,:,:] = image
			self.counter = (self.counter+1)%self.max_count

		if use_cuda: 
			tensor = torch.cuda.FloatTensor(data).float()
		else: 
			tensor = torch.from_numpy(data).float()

		return tensor

loader = Loader()
ae = VAE()

load = True
if load: 
	ae.load_state_dict(torch.load('ae64'))

ae.cuda()

epochs = 0

scheduler = StepLR(ae.adam, step_size = 1000, gamma = 0.8)

mean_loss = 0.
for epoch in range(1,epochs+1): 

	scheduler.step()

	data = loader.sample(batch_size = 32, use_cuda = True)
	pred, mu, std = ae(data)

	loss = ae.eval_loss(pred, data, mu, std)
	ae.maj(loss)

	mean_loss += loss.item()
	if epoch%100 == 0:
		print('Epoch {}/{} -- Mean loss {:.6f} '.format(epoch, epochs, mean_loss/100.))
		mean_loss = 0.

		ae.cpu()
		torch.save(ae.state_dict(), 'ae64')
		ae.cuda()


ae.cpu()

import matplotlib.pyplot as plt 
plt.style.use('ggplot')


while True: 


	sample = ae.demo().detach().numpy()
	sample = sample[0,:,:,:]
	sample = np.transpose(sample, [1,2,0])


	plt.imshow(sample)
	plt.pause(0.1)
	input()
	plt.cla()


	

