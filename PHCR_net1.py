# A deep learning-based progressive method for thick cloud removal from hyperspectral images
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
import numpy as np
from osgeo import gdal
import time
import matplotlib.pyplot as plt
import tifffile as tiff

# Hyper parameters
num_epochs = 120 # defualt=100
batch_size = 256 # 以像元为单位
learning_rate = 0.8 # defualt=1
rd = 20 # 权重衰减轮次
de = 0.8
nei = 5 
band_number = 6
bandsin = nei*nei*band_number # 同时使用9n6b数据
bandsout = 1 # reconstruct one representative band
G = 5 # reconstruct group G5
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

clear = gdal.Open('./Repre5_cloudy.tif') 
known = gdal.Open('./Landsat_auxiliary_25n.tif')
mask = gdal.Open('./mask.tif')

clear = clear.ReadAsArray()
clear = torch.from_numpy(clear.astype(np.float32)).clone()
known = known.ReadAsArray()
known = torch.from_numpy(known.astype(np.float32)).clone()
mask = mask.ReadAsArray()
mask = torch.from_numpy(mask.astype(np.float32)).clone()

Ss = clear.size()
print(Ss)
samp = Ss[1] 
line = Ss[2]
Bs = Ss[0]
sub_size = 1 # pixel-based
w =  int(line/sub_size) # number of patches
h =  int(samp/sub_size)
mask_non = torch.ones(mask.size()) - mask

known_crop = torch.chunk(known*mask_non, w, dim=2) # 3D tensor 
known_crop = torch.cat([fm.unsqueeze(0) for fm in known_crop], dim=0) 
known_crop = torch.chunk(known_crop, h, dim=2) 
known_crop = torch.cat([fm.unsqueeze(0) for fm in known_crop], dim=0) 
subs_known = known_crop.reshape(w*h,bandsin)

knownt_crop = torch.chunk(known, w, dim=2) # 3D tensor 
knownt_crop = torch.cat([fm.unsqueeze(0) for fm in knownt_crop], dim=0) 
knownt_crop = torch.chunk(knownt_crop, h, dim=2) 
knownt_crop = torch.cat([fm.unsqueeze(0) for fm in knownt_crop], dim=0) 
subs_knownt = knownt_crop.reshape(w*h,bandsin)

# AutoEncoder net
class AE(nn.Module):
    def __init__(self, bands=bandsin):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(bandsin, 75)
        self.rl1 = nn.ReLU(True) 

        self.fc2 = nn.Linear(75, 25)
        self.rl2 = nn.ReLU(True)

        self.fc3 = nn.Linear(25, 5)
        self.rl3 = nn.ReLU(True)

        self.fc4 = nn.Linear(5, bandsout)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.rl1(out)
        out = self.fc2(out)
        out = self.rl2(out)
        out = self.fc3(out)
        out = self.rl3(out)
        out = self.fc4(out)
        return out

model = AE(bandsin)

# # Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

# # For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

start_time=time.time()
# ###### Train the model ######
for bn in range(Bs):
    model = model.to(device)
    clear_crop = torch.chunk(clear[bn,:,:], w, dim=1) # 3D tensor 按列分割
    clear_crop = torch.cat([fm.unsqueeze(0) for fm in clear_crop], dim=0) 
    clear_crop = torch.chunk(clear_crop, h, dim=1) 
    clear_crop = torch.cat([fm.unsqueeze(0) for fm in clear_crop], dim=0) 
    subs_clear = clear_crop.reshape(w*h,bandsout)

    Train = TensorDataset(subs_clear,subs_known) # 组合训练数据的特征和标签
    train_iter = DataLoader(dataset=Train, batch_size=batch_size, shuffle=True)

    total_step = len(train_iter)
    print ('mission-->batch size:{}, epoch:{}, step:{}'.format(batch_size, num_epochs, total_step))
    curr_lr = learning_rate
    loss_list = []

    for epoch in range(num_epochs):
        losslist = 0
        for i, (targets,temps) in enumerate(train_iter):
            temps = temps.to(device)
            targets =  targets.to(device)

            # Forward pass
            outputs = model(temps)
            loss = criterion(outputs, targets) # 非云区数据训练
            losslist += loss
            # print(outputs.size(),targets.size())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % num_epochs == 0:
            torch.save(model.state_dict(), './PHCRb{}_{}lr_{}e_{}b_R{}.pth'.format(bn+1,learning_rate,epoch+1, batch_size,G))

        lossmean = losslist/total_step

        if (epoch+1) % 1 == 0:
            print ('Epoch [{}/{}], Loss: {:.8f}, band: {}' 
                    .format(epoch+1, num_epochs, lossmean.item(),bn+1))
        
        loss_list.append(lossmean)

        # Decay learning rate
        if (epoch+1) % rd == 0:
            curr_lr *= de
            update_lr(optimizer, curr_lr)

    ###### predict ######
    print('test data preparing')
    modelt = AE(bandsin)
    modelt.load_state_dict(torch.load('./PHCRb{}_{}lr_{}e_{}b_R{}.pth'.format(bn+1,learning_rate,num_epochs,batch_size,G)))
    pre = modelt(subs_knownt) # S2 to L8 全图
    pre = pre.reshape(samp,-1,bandsout)
    pre = pre.permute(2,0,1) # 维度顺序转换 ##(channel, x, y)
    pre = pre.cpu()  # 将tensor复制到主机中   
    pre = pre.detach().numpy() 
    tiff.imsave('./EOrepre{}pre_{}lr{}e.tif'.format(bn+1,learning_rate,num_epochs),pre) # 输出band1,2,3
    print(bn+1)

print('training time: {:.4f} min'.format((time.time() - start_time)/60))

# pain the loss line chart
x = range(num_epochs)
y = torch.tensor(loss_list, device = 'cpu') # list类型放cpu
plt.subplot(1, 1, 1)
plt.plot(x, y, 'o-')
plt.title('lr:{}_rd:{}'.format(learning_rate, rd))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# python PHCR_net1.py