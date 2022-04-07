import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
import torch
from torch import nn,optim
import torch.nn.functional as F
from load_data import *
import random
from sklearn.metrics import mean_absolute_error
import copy

pred_len = 1
shorttime_history = 12
longtime_periodicity_days = 3

csd = CreataSequentialData(datapath,shorttime_history=shorttime_history,longtime_periodicity_days=longtime_periodicity_days,pred_len=pred_len,channel_last=False,split_ratio=[0.9,0,0.1])
train_dataset = BikeNYCDataset(csd.load_train_data())
test_dataset = BikeNYCDataset(csd.load_test_data())

N = len(train_dataset)
indices = np.arange(N)
np.random.shuffle(indices)
n = int(0.8 * N)
print('{} for training,\t{} for validation'.format(n, N-n))
train_indices = indices[:n]
valid_indices = indices[n:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# train a model
def train(model, dataloader, optimizer, criterion):
    total_loss, total_prediction = 0.0, 0.0
    model.train()
    for X, y in dataloader:
        predictions = model(X)
        loss = criterion(predictions, y.view(-1,12*7))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()*y.shape[0]
        total_prediction += y.shape[0]
    return total_loss / total_prediction

def evaluate(model, dataloader, criterion):  
    total_loss, total_prediction = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            predictions = model(X)
            loss = criterion(predictions, y.view(-1,12*7))

            total_loss += loss.item()*y.shape[0]
            total_prediction += y.shape[0]
    return total_loss / total_prediction


class CNN(nn.Module):
  def __init__(self,input_channel,channel_num):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(input_channel,channel_num[0],kernel_size=3,padding=1)
    self.conv2 = nn.Conv2d(channel_num[0],channel_num[1],kernel_size=3,padding=1)
    self.conv3 = nn.Conv2d(channel_num[1],channel_num[2],kernel_size=3,padding=1)
    self.fc = nn.Linear(channel_num[2]*12*7,12*7)
    self.elu_activate = nn.ELU()
  
  def forward(self, images):
    x = F.relu(self.conv1(images))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = torch.flatten(x, 1)
    #x = self.elu_activate(self.fc(x))
    x = F.tanh(self.fc(x))
    return x


EPOCHS = 20
lr = 1e-3

# call your model here
model = CNN(input_channel=shorttime_history+longtime_periodicity_days,channel_num=[10,5,1])
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
model = model.to(device)
criterion = criterion.to(device)

# train and test the model
# you can reuse the following coding block for hyperparameter tuning
# feel free to try more advanced training strategies
best_valid_loss = np.inf
best_state_dict = copy.deepcopy(model.state_dict())
for epoch in range(EPOCHS):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    valid_loss = evaluate(model, valid_dataloader, criterion)

    print('Epoch {} | Train loss {:.3f} | Valid loss {:.3f}'.format(epoch, train_loss, valid_loss))

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_state_dict = copy.deepcopy(model.state_dict())

# test
def test(model, dataloader, criterion):  
    total_loss, total_prediction = 0.0, 0.0
    model.eval()
    pred_array = []
    with torch.no_grad():
        for X, y in dataloader:
            predictions = model(X)
            loss = criterion(predictions, y.view(-1,12*7))

            total_loss += loss.item()*y.shape[0]
            total_prediction += y.shape[0]
            pred_array.append(predictions.cpu().numpy())
    return total_loss / total_prediction, np.concatenate(pred_array).reshape((-1,12,7))

model.load_state_dict(best_state_dict)
test_loss, pred_output = test(model, test_dataloader, criterion)
print('Test loss {:.3f} '.format(test_loss))


true_demand = csd.minmaxnormalization.inverse_transform(csd.load_test_data()[1])
pred_demand = csd.minmaxnormalization.inverse_transform(pred_output)
MAE_val = mean_absolute_error(true_demand.reshape(-1),pred_demand.reshape(-1))
print('Test MAE {:.3f} '.format(MAE_val))


region_location = (Ellipsis,6,4)
plt.figure(figsize=(8,6))
plt.plot(true_demand[region_location][:120],label='truth')
plt.plot(pred_demand[region_location][:120],label='predict')
plt.legend()
plt.xlabel('time')
plt.ylabel('demand per hour')
plt.savefig('testdemand.png',format='png', transparent=False, dpi=300, pad_inches = 0)
plt.show()