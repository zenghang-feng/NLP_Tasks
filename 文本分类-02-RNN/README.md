# 1. 项目概述

将文本数据用预训练的GloVE向量化，然后采用RNN对IMDB数据进行文本分类；

# 2. 程序实现

```
import torch
import numpy as np
import torch.nn as nn
import torchtext.datasets as datasets
import torchtext.data as data
from torchtext.vocab import GloVe

#################################################################
# 加载数据集，注意torchtext版本，这里用的是 0.5.0
#################################################################
time_steps = 50
TEXT = data.Field(lower=True, batch_first=True, fix_length=time_steps)
LABEL = data.Field(sequential=False)
data_train, data_test = datasets.IMDB.splits(TEXT, LABEL)

max_size = 10000
vec_size = 300
TEXT.build_vocab(data_train, vectors=GloVe(name='6B', dim=vec_size), max_size=max_size - 2, min_freq=10)
LABEL.build_vocab(data_train)

batch_size = 32
train_iter, test_iter = data.BucketIterator.splits((data_train, data_test), batch_size=batch_size, shuffle=True)


#################################################################
# 定义网络参数，网络结构，损失函数，优化器等信息
#################################################################
# 定义网络参数 ===================================================
hide_size = 256  # 自定义向量的维度
out_size = 2  # 是分类标签数量


# 网络结构 =======================================================
class RnnClf(nn.Module):
    def __init__(self, time_steps, batch_size, max_size, vec_size, hide_size, out_size):
        super().__init__()
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.max_size = max_size
        self.vec_size = vec_size
        self.hide_size = hide_size
        self.out_size = out_size

        self.embed = nn.Embedding(self.max_size, self.vec_size)
        self.rnn = nn.RNN(self.vec_size, self.hide_size)
        self.linear = nn.Linear(self.time_steps*self.hide_size, self.out_size)

    def forward(self, x, state_init):

        x_embed = self.embed(x)
        y_rnn, state_rnn = self.rnn(x_embed, state_init)
        y_rnn_p = y_rnn.permute(1, 0, 2)                            # 转置成 batch_size x time_steps x hide_size
        y_rnn_p = y_rnn_p.reshape(y_rnn_p.shape[0], -1)             # 时间步和隐藏维度合并
        y_pre = self.linear(y_rnn_p)

        return y_pre


# 实例化网络 =====================================================
RnnClf = RnnClf(time_steps=time_steps, batch_size=batch_size,
                max_size=max_size, vec_size=vec_size,
                hide_size=hide_size, out_size=out_size)
RnnClf.embed.weight.data = TEXT.vocab.vectors                       # 引入预训练的embedding
RnnClf.embed.weight.requires_grad = False

# 定义优化器 ====================================================
lr = 0.001
optimizer = torch.optim.SGD([p for p in RnnClf.parameters() if p.requires_grad == True], lr=lr)

# 定义损失函数 ===================================================
los_fun = nn.CrossEntropyLoss(reduction="none")

#################################################################
# 定义超参数，训练网络
#################################################################
epochs = 300
print('---------------------- 开始训练模型----------------------')
for epoch in range(epochs):
    loss_all = 0
    for train_sample in train_iter:
        x = train_sample.text                                           # 是一个 batch_size x fix_length 的矩阵(32 x 20)，fix_length是句子长度，相当于时间步
        y = train_sample.label
        x_t = x.t()                                                     # 转置为 时间步 x batch_size

        state_init = torch.zeros(size=(1, x_t.shape[1], hide_size))     # 单隐藏层，所以是 1 x batch_size x hide_size
        y_pre = RnnClf.forward(x_t, state_init)
        loss = los_fun(y_pre, y-1)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        loss_all += loss.mean().data

    print("epoch", epoch, "total loss is", loss_all)
```
