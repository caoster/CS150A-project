import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MinMaxScaler
import math

train_df = pd.read_csv('train_pre.csv', sep='\t')
test_df = pd.read_csv('test_pre.csv', sep='\t')

X = train_df.dropna()
y = np.array(X['Correct First Attempt']).astype(int).ravel()
del X['Correct First Attempt']
XX = test_df.dropna()
yy = np.array(XX['Correct First Attempt']).astype(int).ravel()
del XX['Correct First Attempt']

scalar = MinMaxScaler(feature_range=(0,1))
X = scalar.fit_transform(X)

train_size = int(X.shape[0]*0.8)
test_size = X.shape[0]-train_size
X=np.array(X)
X_train,X_val =X[0:train_size,:],X[train_size:,:]

X_train = torch.Tensor(X_train)
X_val = torch.Tensor(X_val)

y_train=y[:train_size,]
y_val=y[train_size:,]

y_train = torch.Tensor(y_train)
y_val = torch.Tensor(y_val)

#y=y.reshape(-1,1)
XX = torch.Tensor(np.array(XX))
yy = torch.Tensor(yy)


class traindataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index): 
        return X_train[index],y_train[index]
    def __len__(self):
        return  X_train.shape[0]

class valdataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index): 
        return X_val[index],y_val[index]
    def __len__(self):
        return X_val.shape[0]

train_dataset=traindataset()
val_dataset=valdataset()

train_dataloader=DataLoader(train_dataset,batch_size=64)
val_dataloader=DataLoader(val_dataset,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(12,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64,32)
        self.fc6 = nn.Linear(32,16)
        self.fc7 = nn.Linear(16,8)
        self.fc8 = nn.Linear(8,1)
        
    def forward(self,x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = F.tanh(x)
        x = self.fc4(x)
        x = F.tanh(x)
        x = self.fc5(x)
        x = F.tanh(x)
        x = self.fc6(x)
        x = F.tanh(x)
        x = self.fc7(x)
        x = F.tanh(x)
        x = self.fc8(x)
        return x

model = Net()
model=model.cuda()
criterion = nn.MSELoss()
criterion=criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model_path="model_DL.pth"
if model_path:
    model.load_state_dict(torch.load(model_path))
#write=Summ
#model.train()
"""
from tqdm import tqdm
prev_accuracy=0
prev_loss = 9999
epochs = 1000
total_train_step=0
total_test_step=0
for i in tqdm(range(epochs)):
    print("-------第{} 轮训练开始---------".format(i+1))
    for data in train_dataloader:
        row_data,row_y=data
        row_data=row_data.cuda()
        row_y=row_y.cuda()
        row_y=row_y.reshape(-1,1)
        y_pred = model(row_data)
        loss = criterion(y_pred,row_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step=total_train_step+1
        if total_train_step % 100 == 0:
            print("训练次数:{},Loss:{}".format(total_train_step,loss.item()))

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in val_dataloader:
            row_data,row_y=data
            row_data=row_data.cuda()
            row_y=row_y.cuda()
            row_y=row_y.reshape(-1,1)
            y_pred = model(row_data)
            loss = criterion(y_pred,row_y)
            total_test_loss+=loss.item()
            y_pred[y_pred>0.5]=1
            y_pred[y_pred<=0.5]=0
            accuracy=(y_pred==row_y).sum()
            total_accuracy=total_accuracy+accuracy
        if  total_accuracy > prev_accuracy:
            prev_accuracy=total_accuracy
            torch.save(model.state_dict(), "model_DL.pth")  
    print("整体验证集上的Loss:{}".format(total_test_loss))
    total_test_step=total_test_step+1
"""
#torch.save(obj=model.state_dict(), f="model_Net.pth")
#model.eval()
test_y = model.forward(XX.cuda())
test_y=test_y.data.cpu().numpy()
"""
test_ans = []
for t in test_y:
    if t[0] > 0.54:
        test_ans.append(1)
    else:
        test_ans.append(0)
"""
print ('VoteClassifier', np.sqrt(mean_squared_error(yy, test_y)))
