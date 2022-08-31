'''
本程序用来使用
'''
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
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
EPOCH = 2000
BATCH_SIZE = 500
LR = 0.0005  # learning rate

class MyDataSet(Data.Dataset):
    def __init__(self, input_x, input_y):
        self.input_x = input_x
        self.input_y = input_y


    def __len__(self):
        return len(self.input_x)

    def __getitem__(self, idx):
        return self.input_x[idx], self.input_y[idx]
#读取的数据设置
#读取的数据设置
task_name = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
for task_num in range(7):
    task = task_name[task_num]
    for sub in range(10):
        sub_num = sub+1

        #读取任务曲线
        file_design = 'data/design/'+task+'/design.mat'
        train_line = np.loadtxt(file_design,skiprows = 5)
        train_line = train_line[:,::2]
        print(train_line.shape)
        #读取数据集
        train_x_file = 'G:/group_data_result/'+task+'_result/sub_'+str(sub_num)+'/head_av.npy'
        train_y_file = 'G:/group_data_result/'+task+'_result/sub_'+str(sub_num)+'/normalinput2d.npy'
        train_x = np.load(train_x_file)
        train_y = np.load(train_y_file)

        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        print(train_x.shape)#(171276,284)
        print(train_y.shape)#(28546,284)
        print(train_line.shape)


        line_len = train_line.shape[1]
        input_len = train_x.shape[0]
        output_len = train_y.shape[0]
        hidden_len = 256

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
                    nn.Linear(input_len, hidden_len),
                )
                self.decoder = nn.Linear(hidden_len, output_len,bias = False)

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded


        autoencoder = AutoEncoder().to(device)

        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
        loss_func = nn.MSELoss()
        loss_cos = nn.CosineEmbeddingLoss()

        loss_train_all = []
        for epoch in tqdm(range(EPOCH)):
            train_loss = 0
            for train_x, train_y in loader:

                train_x = train_x.to(device)
                train_y = train_y.to(device)
                target_line = torch.ones(line_len)
                target_line = target_line.to(device)

                encoded, decoded = autoencoder(train_x)

                loss = loss_func(decoded, train_y)


                loss_line = loss_cos(encoded.T[0:line_len,:],int_line.T[0:line_len,:],target_line)
                # #测试代码
                # loss_line = loss_cos(encoded.T[0:1,:],int_line.T[0:1,:],target_line)
                # similarity = torch.cosine_similarity(encoded.T[0:1,:],int_line.T[0:1,:], dim=1)
                # print(similarity)
                optimizer.zero_grad()
                #loss.backward()
                #loss_line.backward()
                loss_all = loss + loss_line
                loss_all.backward()

                optimizer.step()


                #print('Epoch: ', epoch, '| mse loss: %.4f' % loss.data.cpu().data.numpy())
                #print('Epoch: ', epoch, '| cos_loss: %.4f' % loss_line.data.cpu().data.numpy())
                #print('Epoch: ', epoch, '| ALL_loss: %.4f' % loss_all.data.cpu().data.numpy())
                #loss_train_all.append(loss.item())
                loss_train_all.append(loss_all.item())
        print('Epoch: ', epoch, '| mse loss: %.4f' % loss.data.cpu().data.numpy())
        print('Epoch: ', epoch, '| cos_loss: %.4f' % loss_line.data.cpu().data.numpy())
        print('Epoch: ', epoch, '| ALL_loss: %.4f' % loss_all.data.cpu().data.numpy())
        encoder_x,encoder_y = autoencoder(int_x)
        print(encoder_x.shape)
        print(encoder_y.shape)
        final_loss = loss_func(encoder_y,int_y)
        print(final_loss.cpu().data.numpy())


        encoder_x = encoder_x.cpu().data.numpy().T
        print(encoder_x.shape)
        np.save('encoder_'+task+'_'+str(sub_num)+'.npy',encoder_x)
        io.savemat('encoder_'+task+'_'+str(sub_num)+'.mat', {'data': encoder_x})
        np.save('loss_train_all_'+task+'_'+str(sub_num)+'.npy',loss_train_all)

        np.save('encoder_result256/encoder_'+task+'_'+str(sub_num)+'.npy',encoder_x)
        io.savemat('encoder_result256/encoder_'+task+'_'+str(sub_num)+'.mat', {'data': encoder_x})
        np.save('encoder_result256/loss_train_all_'+task+'_'+str(sub_num)+'.npy',loss_train_all)

        plt.plot(loss_train_all, 'b')
        plt.legend(["train_loss"])
        plt.grid(True)
        plt.axis('tight')
        plt.xlabel('epoch')
        plt.ylabel('loss_loss')
        plt.show()


