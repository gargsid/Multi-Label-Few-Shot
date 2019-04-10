import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.io
from os import sys
import numpy as np
import random
from matplotlib	 import pyplot as plt 
from sklearn.metrics import classification_report, hamming_loss

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
	n = x.size(0)
	m = y.size(0)
	d = x.size(1)
	assert d == y.size(1)

	x = x.unsqueeze(1).expand(n, m, d)
	y = y.unsqueeze(0).expand(n, m, d)

	return torch.pow(x - y, 2).sum(2)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
	def __init__(self, encoder):
		super(Protonet, self).__init__()

		self.encoder = encoder
	def compute_mean(self,data,length):
		z = self.encoder.forward(data)
		z_dim = z.size(-1)
		z_proto = z.view(1,length,z_dim).mean(1)
		return z_proto
	# def compute_forward(self,data):

	def loss(self, samples, query):
		xs = Variable(torch.tensor(samples).float()) # support [2, 5, 1, 9, 15]
		xq = Variable(torch.tensor(query).float()) # query  [2, 5, 1, 9, 15]


		n_class = 2
		# assert xq.size(0) == n_class
		n_support = 10
		n_query = 10

		target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
		target_inds = Variable(target_inds, requires_grad=False)

		x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
		               xq.view(n_class * n_query, *xq.size()[2:])], 0)

		# print(x.shape)  # [20, 1, 28, 28]
		# sys.exit/()
		z = self.encoder.forward(x)  # [600, 64]
		z_dim = z.size(-1)  # 64   
		z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
		# print(z_proto.shape)  # [60, 64]
		# print(z_proto)
		zq = z[n_class*n_support:]
		# print(zq.shape) [300, 64]

		dists = euclidean_dist(zq, z_proto)
		# print(dists.shape) [300, 60]

		log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
		# print(log_p_y.shape) [60, 5, 60]

		loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

		_, y_hat = log_p_y.max(2)
		acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()


		return loss_val, {
		        'loss': loss_val.item(),
		        'acc': acc_val.item()
		    }

def load_protonet_conv():
	x_dim = [1,28,28]
	hid_dim = 64
	z_dim = 64

	def conv_block(in_channels, out_channels):

		return nn.Sequential(
		    nn.Conv2d(in_channels, out_channels, 3, padding=1),
		    nn.BatchNorm2d(out_channels),
		    nn.Dropout(p=0.2),
		    nn.ReLU(),
		    nn.MaxPool2d(2)
		)

	encoder = nn.Sequential(
		conv_block(x_dim[0], hid_dim),
		conv_block(hid_dim, hid_dim),
		conv_block(hid_dim, hid_dim),
		conv_block(hid_dim, z_dim),
		Flatten()
	)

	return Protonet(encoder)

def eval_metrics(Z, Y):
	# Z = pred, Y = true
	D = len(Z)
	A = np.array(Z&Y.T)
	B = np.array(Z|Y.T)
	a = np.sum(A, axis=1)
	b = np.sum(B, axis=1)
	c = a/b
	acc =  np.sum(c)*1.0/D
	prec = np.sum(a/(np.sum(Z,axis=1)+0.1))*1.0/D
	recall = np.sum(a/np.sum(Y.T,axis=1))*1.0/D

	return acc, prec, recall

learning_rate = 0.001

def train(samples, query, model):
	model.eval()
	loss_val,acc = model.loss(samples, query)
	 # Backward and optimize
	model.train()
	optimizer.zero_grad()
	loss_val.backward()
	optimizer.step()

	return loss_val, acc

n_class = 5
n_samples = 2000
data_path = ('miml_data.mat')
mat = scipy.io.loadmat(data_path)
mat['targets'][mat['targets']==-1] = 0

data = []
for i in range(n_samples):
	data.append(mat['bags'][i][0])
data = np.array(data)

n_train = int(0.8*n_samples)
indices = np.random.choice(n_samples,n_samples,replace = False)
train_data = data[indices[:n_train]]
train_labels = mat['targets'][:,:n_train]
test_data = data[indices[n_train:]]
test_labels = mat['targets'][:,n_train:]


classes = {}
for i in range(n_class):
	classes[i] = {}
for k in range(n_class):
	indices_1 = np.argwhere(train_labels[k]==1)
	indices_0 = np.argwhere(train_labels[k]==0)
	classes[k][0] = data[indices_0]
	classes[k][1] = data[indices_1]
	result = np.zeros((len(classes[k][0]), 1, 28,28))
	result[:,:,:classes[k][0].shape[2],:classes[k][0].shape[3]] = classes[k][0]
	classes[k][0] = result
	result = np.zeros((len(classes[k][1]), 1, 28,28))
	result[:,:,:classes[k][1].shape[2],:classes[k][1].shape[3]] = classes[k][1]
	classes[k][1] = result
	# print('k',k)
	# print(classes[k][0].shape)
	# print(classes[k][1].shape)

# n_epochs = 1
n_epochs = 500
n_batch = 10

for i in range(n_class):
	print("class "+str(i+1))
	model = load_protonet_conv()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)	
	losses = []
	accus = []

	for epoch in range(n_epochs):
		label_1 = np.random.choice(len(classes[i][1]),2*n_batch,replace = False)
		label_0 = np.random.choice(len(classes[i][0]),2*n_batch,replace = False)
		support = np.append([classes[i][1][label_1[:n_batch]]],[classes[i][0][label_0[:n_batch]]],axis = 0)
		# print(support.shape)
		query = np.append([classes[i][1][label_1[n_batch:]]],[classes[i][0][label_0[n_batch:]]],axis = 0)
		result = np.zeros((2,n_batch, 1, 28,28))
		result[:,:,:,:support.shape[3], :support.shape[4]] = support
		support = result
		result[:,:,:,:query.shape[3], :query.shape[4]] = query
		query = result
		# print(query.shape,support.shape)
		loss,acc = train(support, query, model)
		print('Epoch:%d Loss: %f Acc: %f'%(epoch+1, loss, acc['acc']))
		
		if epoch%5==1:
			losses.append(loss)
			accus.append(acc['acc'])

	plt.figure(figsize=(12,8))
	plt.plot(np.arange(len(losses)), np.array(losses), c='red', label='Losses')
	# plt.plot(np.arange(n_epochs), np.array(accus), c='blue', label='Acc')
	plt.xlabel("Episodes")
	plt.ylabel("Classification Loss")
	plt.legend()
	print('Saving Model: %d'%(i+1))
	torch.save(model, 'model_{}.pt'.format(i+1))
	plt.savefig('Class_{0}.png'.format(i+1))

# sys.exit()
model.eval()
# print(test_data.shape)
test_data = np.expand_dims(test_data,axis = 1)
# print(test_data.shape)
pred = np.zeros((len(test_data),n_class))

for i in range(n_class):
	print('Loading Model %d'%(i+1))
	model = torch.load('model_{}.pt'.format(i+1))
	model.eval()
	print('Model Loaded')
	result = np.zeros((len(test_data), 1, 28,28))
	result[:,:,:test_data.shape[2],:test_data.shape[3]] = test_data
	test_data = result
	trans_test = model.encoder.forward(torch.tensor(test_data).float())
	attributes_0 = model.compute_mean(torch.tensor(classes[k][0]).float(),len(classes[k][0]))
	attributes_1 = model.compute_mean(torch.tensor(classes[k][1]).float(),len(classes[k][1]))
	# print(attributes_0.shape)
	t = euclidean_dist(torch.tensor(trans_test).float(),attributes_0)-euclidean_dist(torch.tensor(trans_test).float(),attributes_1)
	print(t.shape)
	t = t.view(400)
	# print(t)
	# print(t.shape)
	# t = t.data.numpy()[:-1]
	# t = np.reshape(t, len(t))
	# print("*********")
	# print(t)
	# print(type(t))
	# print("&*&&&&")
	indices = np.where(t>0)[0]
	# print(indices.shape)
	# print(type(indices))
	# print(indices)
	pred[indices,i] = 1
	# print(pred)

pred = np.array(pred, dtype=int)
acc, prec, recall = eval_metrics(pred, test_labels)
print(acc, prec, recall)
hloss = hamming_loss(pred.ravel(), test_labels.ravel())
print(hloss)
# print(pred)
	# print(result.shape)
	# print(result[0])
