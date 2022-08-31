'''
此程序只为了提取多个人的数据的cls维度是 [6，28546，1，407]

'''

import matplotlib


from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import nilearn
from nilearn.input_data import NiftiMasker
import numpy as np
import nibabel as nib
import numpy as np
import pandas as pd
import re
import math
import torch
import datetime
import numpy as np
from random import *
from random import shuffle
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import requests

def normalization(data):
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range


def make_data(start):
    batch = []
    tokens_a_index = start # sample random index in sentences
    for j in range(1):
        tokens_a = token_list[tokens_a_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1)

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  ## 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # candidate masked position
        cand_maked_pos.remove(1)
        cand_maked_pos.remove(405)
        shuffle(cand_maked_pos)#随机打算mask的程序

        #mask_start = randint(21, 610)

        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]']  # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index  # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        batch.append([input_ids, segment_ids, masked_tokens, masked_pos])

    return batch
def make_predict(start):
    batch_predict = []
    positive = negative = 0
    tokens_a_index = start  # sample random index in sentences

    tokens_a = token_list[tokens_a_index]
    input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']]
    segment_ids = [0] * (1 + len(tokens_a) + 1)

    # MASK LM
    n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  ## 15 % of tokens in one sentence
    cand_maked_pos = [i for i, token in enumerate(input_ids)
                      if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # candidate masked position
    #print(cand_maked_pos)
    mask_start = 40
    del cand_maked_pos[39:59]
    remove_mask1 = []
    remove_mask2 = []
    for re in range(20):
        remove_mask1.append(mask_start+re)
        remove_mask2.append(mask_start + 20 + re * 8)
        cand_maked_pos.remove(mask_start + 20 + re * 8)

    shuffle(remove_mask2)
    cand_maked_pos.remove(1)
    cand_maked_pos.remove(2)
    cand_maked_pos.remove(3)
    cand_maked_pos.remove(405)
    #shuffle(cand_maked_pos)#随机打算mask的程序
    cand_maked_pos = remove_mask1+remove_mask2+cand_maked_pos

    #mask_start = randint(21, 260)

    masked_tokens, masked_pos = [], []
    for pos in cand_maked_pos[:n_pred]:
        masked_pos.append(pos)
        masked_tokens.append(input_ids[pos])
        if random() < 0.8:  # 80%
            input_ids[pos] = word2idx['[MASK]']  # make mask
        elif random() > 0.9:  # 10%
            index = randint(0, vocab_size - 1)  # random index in vocabulary
            while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                index = randint(0, vocab_size - 1)
            input_ids[pos] = index  # replace

    # Zero Paddings
    n_pad = maxlen - len(input_ids)
    input_ids.extend([0] * n_pad)
    segment_ids.extend([0] * n_pad)

    # Zero Padding (100% - 15%) tokens
    if max_pred > n_pred:
        n_pad = max_pred - n_pred
        masked_tokens.extend([0] * n_pad)
        masked_pos.extend([0] * n_pad)

    batch_predict.append([input_ids, segment_ids, masked_tokens, masked_pos])

    return batch_predict


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.device = device
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        pos = pos.to(device)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        embedding = embedding.to(device)

        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.device = device

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        attv = context

        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # context: [batch_size, seq_len, n_heads, d_v]
        output = self.linear(context)
        attliner = output
        out = self.LayerNorm(output + residual)

        # return nn.LayerNorm(d_model)(output + residual)  # output: [batch_size, seq_len, d_model]
        return out, attn,attv,attliner


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.device = device

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        a = self.fc1(x)
        b = gelu(a)
        c = self.fc2(b)
        # return self.fc2(gelu(self.fc1(x)))
        return c


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.device = device

        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn,attv,attliner = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        attention_outputs = enc_outputs
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]

        return enc_outputs, attention_outputs, attn,attv,attliner


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.device = device
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, attention_output, enc_self_attn,attv,attliner = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))  # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]\

        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        logits_lm = self.decoder(h_masked) + self.decoder_bias  # [batch_size, max_pred, n_vocab]
        return logits_lm, logits_clsf, h_pooled, output, attention_output, enc_self_attn,attv,attliner

if __name__ == '__main__':
    # encoding=utf8
    '''
    bert模型超参数设置
    '''
    maxlen = 407
    predictsize = 100 #预测集抽取的样本数
    epochsize = 600
    batch_size = 100
    max_pred = 30  # max tokens of prediction
    n_layers = 3
    n_heads = 6
    d_model = 400
    d_ff = d_model * 4  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2

    '''
    查看和显示nii文件
    '''
    mask_imgback = nib.load('/root/autodl-tmp/WM_test/ADHD200_mask_152_4mm.nii.gz')
    # 这是把四维nii数据转成二维数据的代码
    masker = NiftiMasker(mask_img=mask_imgback)
    masker.fit()
    inputdata2d = np.load('/root/autodl-tmp/WM_test/data/WM_sub_8.npy')
    print(inputdata2d.shape)
    '''
    将input2d数据进行预处理，归一化之后保留3位有效数字，变量为normalinput2d
    '''
    data = inputdata2d.T
    data1=normalization(data)
    normalinput2d = np.round(data1, 3)
    np.save("/root/autodl-tmp/WM_test/result/sub_8/normalinput2d.npy", normalinput2d)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    start = datetime.datetime.now()

    three_list = []
    list_test1 = normalinput2d.tolist()
    sentences = list_test1
    print(len(list_test1))
    list_test2 = normalinput2d.reshape(405*28546).tolist()
    word_list = list(set(list_test2))

    word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        word2idx[w] = i + 4

    idx2word = {i: w for i, w in enumerate(word2idx)}
    vocab_size = len(word2idx)

    token_list = list()
    for sentence in range(normalinput2d.shape[0]):
        arr = [word2idx[s] for s in normalinput2d[sentence, :]]
        token_list.append(arr)
    '''
    训练集数据制作
    '''
    # Proprecessing Finished
    train_list = range(28546)
    train_list = [i for i in train_list]

    pre_list = range(2800)
    pre_list = [i * 10 for i in pre_list]
 
    batch = []
    for m in train_list:
        batch.extend(make_data(m))

    input_ids, segment_ids, masked_tokens, masked_pos = zip(*batch)
    input_ids, segment_ids, masked_tokens, masked_pos = torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), torch.LongTensor(masked_pos)
    loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos), batch_size)
    '''
    预测集数据制作
    '''
    batch_predict = []
    for n in train_list:
        batch_predict.extend(make_data(n))
    input_ids_pre, segment_ids_pre, masked_tokens_pre, masked_pos_pre = zip(*batch_predict)
    input_ids_pre, segment_ids_pre, masked_tokens_pre, masked_pos_pre = torch.LongTensor(input_ids_pre), \
                                                                        torch.LongTensor(segment_ids_pre), \
                                                                        torch.LongTensor(masked_tokens_pre), \
                                                                        torch.LongTensor(masked_pos_pre)
    loader_pre = Data.DataLoader(MyDataSet(input_ids_pre, segment_ids_pre, masked_tokens_pre, masked_pos_pre), batch_size)

    '''
    Bert模型搭建
    '''
    model = BERT().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adamax(model.parameters(), lr=0.0002)
    '''
     开始训练数据
    '''
    from tqdm import tqdm

    model.train()
    loss_train_all = []
    #loss_val_all = []
    start_train = datetime.datetime.now()
    for epoch in range(epochsize):
        i = 0
        train_loss = 0
        for input_ids, segment_ids, masked_tokens, masked_pos in loader:
            i += 1
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            masked_tokens = masked_tokens.to(device)
            masked_pos = masked_pos.to(device)

            logits_lm, logits_clsf, h_pooled, output, attention_output,enc_self_attn,attv,attliner = model(input_ids, segment_ids, masked_pos)

            loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))  # for masked LM
            loss_lm = (loss_lm.float()).mean()
            loss = loss_lm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / i
        loss_train_all.append(train_loss)
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(train_loss))
    end = datetime.datetime.now()
    print("训练时间为")
    print(end-start_train)

    '''
     预测完成后对28546个点输出attentionmap值
    '''
    input_ids1, segment_ids1, masked_tokens1, masked_pos1= make_data(0)[0]
    logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1,enc_self_attn1,attv1,attliner1= model(torch.LongTensor([input_ids1]).to(device),
                                                                           torch.LongTensor([segment_ids1]).to(device),
                                                                           torch.LongTensor([masked_pos1]).to(device))

    enc_self_attn11 = enc_self_attn1.cpu().data.numpy()
    print(enc_self_attn11.shape)
    from tqdm import tqdm
    for j in range(57):
        del input_ids1, segment_ids1, masked_tokens1, masked_pos1
        del logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1, enc_self_attn1,attv1,attliner1,enc_self_attn11
        input_ids1, segment_ids1, masked_tokens1, masked_pos1 = make_data(j*500)[0]
        print(j*500)

        logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1, enc_self_attn1,attv1,attliner1 = model(
            torch.LongTensor([input_ids1]).to(device),
            torch.LongTensor([segment_ids1]).to(device),
            torch.LongTensor([masked_pos1]).to(device))

        enc_self_attn11 = enc_self_attn1.cpu().data.numpy()
        for a_map01 in tqdm(range(499)):
            input_ids1, segment_ids1, masked_tokens1, masked_pos1= make_data(a_map01+j*500+1)[0]


            logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1,enc_self_attn1,attv1,attliner1= model(torch.LongTensor([input_ids1]).to(device),
                                                                               torch.LongTensor([segment_ids1]).to(device),
                                                                               torch.LongTensor([masked_pos1]).to(device))
            enc_self_attn01 = enc_self_attn1.cpu().data.numpy()
            enc_self_attn11= np.concatenate((enc_self_attn11,enc_self_attn01),axis = 0)
        np.save('/root/autodl-tmp/WM_test/result/sub_8/enc_self_attn_cls' + str(j) +'.npy',enc_self_attn11[:,:,0,:])
        print(enc_self_attn11.shape)

    del input_ids1, segment_ids1, masked_tokens1, masked_pos1
    del logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1, enc_self_attn1,attv1,attliner1,enc_self_attn11
    input_ids1, segment_ids1, masked_tokens1, masked_pos1= make_data(28500)[0]
    logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1,enc_self_attn1,attv1,attliner1= model(torch.LongTensor([input_ids1]).to(device),
                                                                           torch.LongTensor([segment_ids1]).to(device),
                                                                           torch.LongTensor([masked_pos1]).to(device))

    enc_self_attn11 = enc_self_attn1.cpu().data.numpy()
    print(enc_self_attn11.shape)
    for a_map01 in tqdm(range(45)):
        input_ids1, segment_ids1, masked_tokens1, masked_pos1= make_data(a_map01+28501)[0]

        logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1,enc_self_attn1,attv1,attliner1= model(torch.LongTensor([input_ids1]).to(device),
                                                                           torch.LongTensor([segment_ids1]).to(device),
                                                                           torch.LongTensor([masked_pos1]).to(device))
        enc_self_attn01 = enc_self_attn1.cpu().data.numpy()
        enc_self_attn11= np.concatenate((enc_self_attn11,enc_self_attn01),axis = 0)

    np.save('/root/autodl-tmp/WM_test/result/sub_8/enc_self_attn_cls57.npy',enc_self_attn11[:,:,0,:])
    print(enc_self_attn11.shape)

'''
定义滑动平均的函数
'''

#批量进行滑动平均
def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re



'''
读取已经训练好的数据，并且将其综合为一个数据28546,6,407
'''
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1),copy=True, clip=False)

import numpy as np
from tqdm import tqdm

enc_self_attn_final = np.load('/root/autodl-tmp/WM_test/result/sub_8/enc_self_attn_cls' + str(0) +'.npy')
for j in tqdm(range(57)):
    enc_self_attn0 = np.load('/root/autodl-tmp/WM_test/result/sub_8/enc_self_attn_cls' + str(j+1) +'.npy')
    enc_self_attn_final = np.concatenate((enc_self_attn_final, enc_self_attn0), axis=0)
    del enc_self_attn0
av_data = enc_self_attn_final.reshape(28546 * 6, 407)
np.save('/root/autodl-tmp/WM_test/result/sub_8/2d_data_407.npy', av_data)
print(av_data.shape)
av_data = av_data[:, 1:406]
np.save('/root/autodl-tmp/WM_test/result/sub_8/2d_data_405.npy', av_data)
print(av_data.shape)
for i in range(6):
    out = moving_average(enc_self_attn_final[0,i,1:406], 10)
    out = moving_average(out, 10)
    out = moving_average(out, 10)
    out = out.reshape(1, -1)
    for j in tqdm(range(28545)):
        y_av = moving_average(enc_self_attn_final[j+1,i,1:406], 10)
        y_av = moving_average(y_av, 10)
        y_av = moving_average(y_av, 10)
        y_av = y_av.reshape(1, -1)
        out = np.concatenate((out, y_av), axis=0)

    print(out.shape)
    np.save('/root/autodl-tmp/WM_test/result/sub_8/out_av_head'+str(i)+'.npy', out)
    del out,y_av
np.save('/root/autodl-tmp/WM_test/result/sub_8/enc_self_attn_cls_final.npy', enc_self_attn_final)
import numpy as np
file_A = '/root/autodl-tmp/WM_test/result/sub_8/2d_data_405.npy'
file_0 = '/root/autodl-tmp/WM_test/result/sub_8/out_av_head0.npy'
file_1 = '/root/autodl-tmp/WM_test/result/sub_8/out_av_head1.npy'
file_2 = '/root/autodl-tmp/WM_test/result/sub_8/out_av_head2.npy'
file_3 = '/root/autodl-tmp/WM_test/result/sub_8/out_av_head3.npy'
file_4 = '/root/autodl-tmp/WM_test/result/sub_8/out_av_head4.npy'
file_5 = '/root/autodl-tmp/WM_test/result/sub_8/out_av_head5.npy'
file_B = '/root/autodl-tmp/WM_test/result/sub_8/enc_self_attn_cls_final.npy'

A = np.load(file_A)
head_0 = np.load(file_0)
head_1 = np.load(file_1)
head_2 = np.load(file_2)
head_3 = np.load(file_3)
head_4 = np.load(file_4)
head_5 = np.load(file_5)

B = np.load(file_B)


head_av = np.concatenate((head_0,head_1),axis = 0)
head_av = np.concatenate((head_av,head_2),axis = 0)
head_av = np.concatenate((head_av,head_3),axis = 0)
head_av = np.concatenate((head_av,head_4),axis = 0)
head_av = np.concatenate((head_av,head_5),axis = 0)
print(head_av.shape)
np.save('/root/autodl-tmp/WM_test/result/sub_8/head_av.npy',head_av)
import scipy.io as io

io.savemat('/root/autodl-tmp/WM_test/result/sub_8/2d_data_405.mat', {'data232': A})
io.savemat('/root/autodl-tmp/WM_test/result/sub_8/head_avWM.mat', {'head_avWM': head_av})

#删除部分中间结果
import os
print('删除部分中间结果')
for de in tqdm(range(58)):
    if os.path.exists('/root/autodl-tmp/WM_test/result/sub_8/enc_self_attn_cls' + str(de) +'.npy'):
        os.remove('/root/autodl-tmp/WM_test/result/sub_8/enc_self_attn_cls' + str(de) +'.npy')
    else:
        print("The file does not exist")

print(end-start_train)



