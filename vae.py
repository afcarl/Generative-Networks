import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim	 
from torch.autograd import Variable
from torch.distributions import Normal 

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST as mnist
from torchvision import transforms

import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

batch_size = 32 

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
data = DataLoader(mnist('../data', train = True, download = False, transform = transform), shuffle = True, batch_size = batch_size)


class VAE(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		self.e = nn.Linear(784,400)
		self.mu = nn.Linear(400,30)
		self.std = nn.Linear(400,30)

		self.d = nn.Linear(30,400)
		self.out = nn.Linear(400,784)

		self.adam = optim.Adam(self.parameters(),1e-3)

	def encode(self,x): 

		x = F.relu(self.e(x))
		mu = self.mu(x)
		std = self.std(x)

		return mu, std

	def sample(self, mu, logvar): 

		eps = torch.FloatTensor(logvar.size()).normal_()
		eps = Variable(eps).cuda()
		z = mu + torch.exp(logvar/2.)*eps
		return z 

	def demo(self): 

		m = torch.zeros((1,30))
		stds = torch.ones_like(m)

		dist = Normal(m,stds)
		z = dist.sample()

		return self.decode(Variable(z))

	def decode(self, z): 

		x = F.relu(self.d(z))
		out = F.sigmoid(self.out(x))

		return out

	def forward(self, x): 

		mu, logvar = self.encode(x)
		z = self.sample(mu, logvar)
		pred = self.decode(z)

		return pred, mu, logvar

	def update(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		self.adam.step()

train = True
if train: 

	epochs = 5
	vae = VAE()
	vae.cuda()
	log = 1000

	for epoch in range(epochs): 

		for i, (x,y) in enumerate(data): 


			vx = Variable(x.view(batch_size,-1)).cuda()
			pred, mu, logvar = vae(vx)
			# Recon loss 

			r_loss = F.mse_loss(pred, Variable(vx.data).cuda())

			# kl divergence  
			kl_loss = 1 + logvar -(mu**2 + torch.exp(logvar))
			kl_loss =-0.5*torch.sum(kl_loss)


			loss = r_loss + kl_loss
			vae.update(loss)
			
			if i%log: 
					print('Epoch {}, Batch {}/{} ({:.0f}%) \t Loss {} '.format(epoch, i*x.shape[0], len(data.dataset), 
						100.*i*x.shape[0]/len(data.dataset), loss.cpu().data[0]))


vae.cpu()
torch.save(vae, 'vae.mnist')

#-------------------------------
#			TEST 
#-------------------------------


vae = torch.load('vae.mnist')

for i in range(100): 

	val = vae.demo().data.numpy().reshape(28,28)
	plt.imshow(val, cmap ='gray')
	plt.pause(0.1)
	input()
	plt.cla()
