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
from torch.autograd import Variable

class GruRNN(nn.Module):

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
 
        self.gru = nn.GRU(input_size, hidden_size, num_layers)  
        self.linear1 = nn.Linear(hidden_size, 16) 
        self.linear2 = nn.Linear(16, output_size) 
 
    def forward(self, _x):
        _x=_x.to(torch.float32)
        x, _ = self.gru(_x)  
        s, b, h = x.shape  
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(s, b, -1)
        return x


device = torch.device("cpu")

if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

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
X=np.array(X)
XX=np.array(XX)
X_train,y_train=X[:train_size,],y[:train_size,]
X_val,y_val=X[train_size:,],y[train_size:,]
X_test,y_test=XX,yy

INPUT_FEATURES_NUM = 12
OUTPUT_FEATURES_NUM = 1
train_x_tensor = X_train.reshape(-1, 1, INPUT_FEATURES_NUM)  
train_y_tensor = y_train.reshape(-1, 1, OUTPUT_FEATURES_NUM)  
val_x_tensor = X_val.reshape(-1, 1, INPUT_FEATURES_NUM)  
val_y_tensor = y_val.reshape(-1, 1, OUTPUT_FEATURES_NUM)

train_x_tensor = torch.from_numpy(train_x_tensor)
train_y_tensor = torch.from_numpy(train_y_tensor)
val_x_tensor = torch.from_numpy(val_x_tensor)
val_y_tensor = torch.from_numpy(val_y_tensor)

"""
X_train,X_val =X[0:train_size,:],X[train_size:,:]
y_train=y[:train_size,]
y_val=y[train_size:,]

XX = torch.Tensor(np.array(XX))
yy = torch.Tensor(yy)
"""

class traindataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index): 
        return train_x_tensor[index],train_y_tensor[index]
    def __len__(self):
        return  train_x_tensor.shape[0]

class valdataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index): 
        return val_x_tensor[index],val_y_tensor[index]
    def __len__(self):
        return val_x_tensor.shape[0]

train_dataset=traindataset()
val_dataset=valdataset()

train_dataloader=DataLoader(train_dataset,batch_size=64)
val_dataloader=DataLoader(val_dataset,batch_size=64)

gru_model = GruRNN(INPUT_FEATURES_NUM, 50, output_size=OUTPUT_FEATURES_NUM, num_layers=10).to(device) 
gru_model=gru_model.cuda()
model_path="model_DL_pytorch.pth"
if model_path:
    gru_model.load_state_dict(torch.load(model_path)) 
print('GRU model:', gru_model )
print('model.parameters:', gru_model.parameters)
print('train x tensor dimension:', Variable(train_x_tensor).size())

criterion = nn.MSELoss()
criterion=criterion.cuda()
optimizer = torch.optim.Adam(gru_model .parameters(), lr=1e-2)

prev_loss = 9999
max_epochs = 2000
total_train_step=0
total_test_step=0
"""
train_x_tensor = train_x_tensor.to(device)
train_y_tensor = train_y_tensor.to(device)
train_y_tensor = train_y_tensor.to(float)

from tqdm import tqdm
for epoch in tqdm(range(max_epochs)):
    print("-------第{} 轮训练开始---------".format(epoch+1))
    for data in train_dataloader:
        row_data,row_y=data
        row_data=row_data.cuda()
        row_y=row_y.cuda()
        row_y=row_y.to(float)
        row_y=row_y.reshape(-1,1,1)
        output = gru_model(row_data)
        output = output.to(float)
        loss = criterion(output, row_y)

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
            row_y=row_y.reshape(-1,1,1)
            y_pred = gru_model(row_data)
            loss = criterion(y_pred,row_y)
            total_test_loss+=loss.item()
            y_pred[y_pred>0.5]=1
            y_pred[y_pred<=0.5]=0
            accuracy=(y_pred==row_y).sum()
            total_accuracy=total_accuracy+accuracy
        if  total_test_loss < prev_loss:
            prev_loss=total_test_loss
            torch.save(gru_model.state_dict(), "model_DL_pytorch.pth")  
    print("整体验证集上的Loss:{}".format(total_test_loss))
    total_test_step=total_test_step+1
"""
gru_model = gru_model.eval()  

test_x_tensor =X_test.reshape(-1, 1,INPUT_FEATURES_NUM)
y_test = y_test.reshape(-1, OUTPUT_FEATURES_NUM) 

test_x_tensor = torch.from_numpy(test_x_tensor)  
test_x_tensor = test_x_tensor.to(device)

pred_y_for_test = gru_model(test_x_tensor).to(device)
pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()

#pred_y_for_test[pred_y_for_test>0.5]=1
#pred_y_for_test[pred_y_for_test<=0.5]=0
print ('VoteClassifier', np.sqrt(mean_squared_error(pred_y_for_test, y_test)))


