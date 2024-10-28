# 1 项目概述  

项目所用的数据集是商品评论的数据集，每一个评论样本对应正面评论或者负面评论的标签，数据集地址如下：  

https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/online_shopping_10_cats.zip

当前是项目第一个版本，主要思路是：  

1、将文本数据分词、去停用词；  

2、然后将文本数据用词袋模型向量化；  

3、最后采用多层感知机对文本情感进行分类；  

# 2 程序实现

```
import re
import jieba
import torch
import numpy as np
import pandas as pd
import torch.utils.data as Data
from torch import nn


#################################################################
# 读取数据，预处理数据
#################################################################
# 数据集包含3列：
# cat 是商品所属的 种类；
# label 是商品评论的好坏（1代表正面、0代表负面）；
# review 是商品评论的文本内容
df = pd.read_csv("online_shopping_10_cats.csv")
df = df.iloc[0:10000, :]

# 对评论文本进行分词 ==============================================
df['review_cut'] = df.apply(lambda row:
                            jieba.lcut(re.sub(r'[\W]', '', str(row['review']))),
                            axis=1)

# 将每个词汇打上序号（序号从0开始）==================================
list_review_c = df['review_cut'].to_list()
word_index = {}
index_tmp = 0
for rl in list_review_c:
    for w in rl:                                        # 遍历每条分此后的词汇列表
        if w not in word_index:                         # 遍历列表中每个词汇
            word_index[w] = index_tmp
            index_tmp += 1

# 采用词袋模型对文本数据向量化 =====================================
words_count = len(word_index)
data_x = []
for rl in list_review_c:
    x_tmp = np.zeros(shape=words_count,  dtype=int)
    for w in rl:
        index_tmp = word_index[w]
        x_tmp[index_tmp] = 1
    data_x.append(x_tmp)
    
data_y = df['label'].to_list()
data_x = np.array(data_x)
data_y = np.array(data_y)

#################################################################
# 构建分类数据集
#################################################################
x_train = torch.tensor(data_x[:-3000], dtype=torch.float)
y_train = torch.tensor(data_y[:-3000], dtype=torch.long)
x_test = torch.tensor(data_x[-3000:], dtype=torch.float)
y_test = torch.tensor(data_y[-3000:], dtype=torch.long)

# 将训练数据的特征和标签组合 =========================================
data_train = Data.TensorDataset(x_train, y_train)
data_test = Data.TensorDataset(x_test, y_test)

print('------------------数据准备完成----------------------------')
# 随机读取小批量 ==================================================
batch_size = 128
train_iter = Data.DataLoader(data_train, batch_size, shuffle=True)
test_iter = Data.DataLoader(data_test, batch_size)

#################################################################
# 构建MLP模型
#################################################################
# 定义网络结构 ====================================================
mlp_model = nn.Sequential(
    nn.Linear(words_count, 128),
    nn.ReLU(),
    nn.Linear(128,2))

# 定义损失函数 ====================================================
loss = nn.CrossEntropyLoss(reduction='none')

# 定义优化器 ======================================================
lr = 0.01
optimizer = torch.optim.SGD(mlp_model.parameters(), lr=lr)

# 训练模型 ========================================================
epochs = 1000
for epoch in range(epochs):
    loss_all = torch.tensor(0, dtype=float)
    for x, y in train_iter:
        y_pre = mlp_model(x)                # 前向计算
        loss_tmp = loss(y_pre, y)           # 计算交叉熵损失，输出是一个 batch_size x 1 的向量
        optimizer.zero_grad()               # 梯度清零
        loss_tmp.mean().backward()                 # 反向传播计算梯度
        optimizer.step()                    # 更新参数

        loss_all += loss_tmp.mean()
    print("epoch", epoch, "total loss is", loss_all)

print('--------------------训练完成-----------------------')


################################################################
# 模型预测
################################################################
total_correct = 0
for x_test, y_test in test_iter:
    y_pre = mlp_model(x_test)
    pred = torch.argmax(y_pre, dim=1)
    correct = torch.as_tensor(torch.eq(pred, y_test), dtype=torch.int64)
    correct = torch.sum(correct)
    total_correct += correct

acc = total_correct / len(test_iter.dataset)
print('test_acc', acc)

```

按照当前参数训练完成之后，在测试集上的准确率为70%多，后续再优化一下。
