
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import scipy.io as io
from nilearn.input_data import NiftiMasker
import nibabel as nib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
EPOCH = 2000

BATCH_SIZE = 284
LR = 0.0005  # learning rate

class MyDataSet(Data.Dataset):
    def __init__(self, input_x, input_y):
        self.input_x = input_x
        self.input_y = input_y


    def __len__(self):
        return len(self.input_x)

    def __getitem__(self, idx):
        return self.input_x[idx], self.input_y[idx]
train_x = np.load('data/cls_284av.npy')
train_y = np.load('data/normalinput2d.npy')

img = nib.load('data/tfMRI_MOTOR_LR.nii.gz')
masker = NiftiMasker()
masked_data = masker.fit_transform(img).T  
def normalization(data):
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range
train_line = np.load('data/MOTOR_taskdesign_1.npy')
print(train_x.shape)
print(train_y.shape)
print(train_line.shape)

int_x,int_y,int_line = torch.Tensor(train_x.T), torch.Tensor(train_y.T),torch.Tensor(train_line),
int_x = int_x.to(device)
int_y = int_y.to(device)
int_line = int_line.to(device)

loader = Data.DataLoader(MyDataSet(int_x, int_y), BATCH_SIZE)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(204354, 50),
        )
        self.decoder = nn.Linear(50, 34059,bias = False)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder().to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()
loss_cos = nn.CosineEmbeddingLoss()

loss_train_all = []
for epoch in range(EPOCH):
    train_loss = 0
    for train_x, train_y in loader:

        train_x = train_x.to(device)
        train_y = train_y.to(device)
        target_line = torch.ones(12)
        target_line = target_line.to(device)

        encoded, decoded = autoencoder(train_x)

        loss = loss_func(decoded, train_y)


        loss_line = loss_cos(encoded.T[0:12,:],int_line.T[0:12,:],target_line)
        optimizer.zero_grad()

        loss_all = loss + loss_line*10
        loss_all.backward()

        optimizer.step()


        print('Epoch: ', epoch, '| cos_loss: %.4f' % loss_line.data.cpu().data.numpy())
        print('Epoch: ', epoch, '| ALL_loss: %.4f' % loss_all.data.cpu().data.numpy())
        loss_train_all.append(loss_line.item())
torch.save(autoencoder, 'autoencoder_100.pkl')
torch.save(autoencoder, 'autoencoder_100.pth')

encoder_x,encoder_y = autoencoder(int_x)

final_loss = loss_func(encoder_y,int_y)
print(final_loss.cpu().data.numpy())


encoder_x = encoder_x.cpu().data.numpy().T
print(encoder_x.shape)
np.save('encoder_x100.npy',encoder_x)
io.savemat('encoder_x100.mat', {'data': encoder_x})


plt.plot(loss_train_all, 'b')
plt.legend(["train_loss"])
plt.grid(True)
plt.axis('tight')
plt.xlabel('epoch')
plt.ylabel('loss_loss')
plt.show()


