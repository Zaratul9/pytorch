import os 
import cv2
import numpy as np 
import matplotlib as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

REBUILD_DATA = False
IMG_SIZE = 50
HALF_IMG_SIZE = round(IMG_SIZE/2)
class DogsVSCats():
	img_size = IMG_SIZE
	half_size = HALF_IMG_SIZE
	CATS = 'PetImages/Cat'
	DOGS = 'PetImages/Dog'
	LABELS = {CATS: 0, DOGS: 1}
	training_data = []
	total = 0
	print(img_size)
	print(half_size)
	def make_training_data(self):
		for label in self.LABELS:
			print(label)
			for f in tqdm(os.listdir(label)):
				if "jpg" in f:
					try:
						path = os.path.join(label, f)
						img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
						full_img = cv2.resize(img, (self.img_size,self.img_size))
						
						half_img = cv2.resize(img, (self.half_size,self.half_size))

						self.training_data.append([np.array(half_img), np.array(full_img)])
						self.total += 1
					except Exception as e:
						print("failed")
						pass
		np.random.shuffle(self.training_data)
		np.save("DLSS_train.npy", self.training_data)
		print("Total:", self.total)


if REBUILD_DATA:
	build = DogsVSCats()
	build.make_training_data()


training_data = np.load("DLSS_train.npy", allow_pickle = True)


class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, 3)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.conv3 = nn.Conv2d(64, 128, 3)

		x = torch.rand(HALF_IMG_SIZE, HALF_IMG_SIZE).view(-1, 1, HALF_IMG_SIZE, HALF_IMG_SIZE)
		self._to_linear = None
		self.convs(x)

		self.fc1 = nn.Linear(self._to_linear, 512)
		self.fc2 = nn.Linear(512, IMG_SIZE, IMG_SIZE)

	def convs(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
		x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))

		#print(x[0].shape)

		if self._to_linear is None:
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
		return x


	def forward(self, x):
		x = self.convs(x)
		x = x.view(-1, self._to_linear)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x, dim=1)

net = Net()


X = torch.Tensor([i[0] for i in training_data]).view(-1, HALF_IMG_SIZE, HALF_IMG_SIZE)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data]).view(-1, IMG_SIZE, IMG_SIZE)

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X))
print(len(test_X))


BATCH_SIZE = 10

EPOCHS = 1

for epoch in range(EPOCHS):
	for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
		#print(i, i+BATCH_SIZE)

		batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,HALF_IMG_SIZE, HALF_IMG_SIZE)
		batch_y = train_y[i:i+BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)

		net.zero_grad()
		outputs = net(batch_X)
		loss = loss_function(outputs, batch_y)
		loss.backward()
		optimizer.step()

print(loss)
