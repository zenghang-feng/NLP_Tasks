import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

#################################################################
# 读取数据，预处理数据
#################################################################
# 按照每一行读取全部的文本到一个列表 ==================================
with open(file="./data/train.txt", mode="r", encoding="utf8") as f:
    list_txt_line = []
    line = f.readline()
    while line:
        list_txt_line.append(line)
        line = f.readline()

# 提取每一个样本序列（句子）到一个列表中，文本字符和对应的标签按照顺序存储 ===
list_x, list_y, list_x_tmp, list_y_tmp = [], [], [], []
dit_x, dit_y = {}, {}
id_x, id_y = 0, 0
len_max = 0
for l in list_txt_line:
    if l == "\n":
        list_x.append(list_x_tmp)
        list_y.append(list_y_tmp)
        # 获取样本数据中最大序列长度，后续按照最大长度对较短的文本填充 -----
        if len(list_x_tmp) > len_max:
            len_max = len(list_x_tmp)
        list_x_tmp = []
        list_y_tmp = []
        continue

    l = l.replace("\n", "")
    x = l[0]
    list_x_tmp.append(x)
    # 将样本数据中每个中文字符设置唯一编号，存储到字典中 -----------------
    if x not in dit_x:
        dit_x[x] = id_x
        id_x += 1

    y = l[2:]
    list_y_tmp.append(l[2:])
    # 将样本数据中每个中文字符对应的NER标签设置唯一编号,存储到字典中 ------
    if y not in dit_y:
        dit_y[y] = id_y
        id_y += 1

# 添加填充字符到字符的字典中 ！！！====================================
dit_x["PAD"] = id_x

# 将字符和对应的标签转换为数字编号，后续输入模型进行计算 ==================
list_x_id, list_y_id = [], []
len_xy = len(list_x)
for i in range(len_xy):
    list_x_tmp_id, list_y_tmp_id = [], []
    len_xy_sub = len(list_x[i])
    for j in range(len_max):
        if j < len_xy_sub:
            list_x_tmp_id.append(dit_x[list_x[i][j]])
            list_y_tmp_id.append(dit_y[list_y[i][j]])
        # 按照最长样本序列对较短序列进行填充 --------------------------
        else:
            list_x_tmp_id.append(dit_x["PAD"])
            list_y_tmp_id.append(dit_y["O"])
    list_x_id.append(list_x_tmp_id)
    list_y_id.append(list_y_tmp_id)

#################################################################
# 将样本数据中的字符和标签对应组合为数据集，
#################################################################
# 转换成tensor数据集 ==============================================
tensor_x = torch.tensor(np.array(list_x_id))
tensor_y = torch.tensor(np.array(list_y_id), dtype=torch.long)
data_xy = Data.TensorDataset(tensor_x, tensor_y)
# 批量加载数据 ====================================================
batch_size = 64
data_iter = Data.DataLoader(dataset=data_xy, batch_size=batch_size, shuffle=True, drop_last=True)

#################################################################
# 构建网络模型，定义损失函数，定义优化器
#################################################################
# 定义网络模型 ====================================================
vocab_size = len(dit_x)
embed_size = 256
hidden_size = 128
output_size = len(dit_y)


class LstmModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(LstmModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 嵌入层，将字符根据序号向量化 ====================
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        # 单层单向RNN =================================
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, batch_first=True)
        # 线性层，输出每个时间步的预测值 ==================
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):
        h, c = self.init_state(x.shape[0])
        out_embed = self.embed(x)
        out_lstm, (out_h, out_c) = self.lstm(out_embed, (h, c))
        out_linear = self.linear(out_lstm.reshape(-1, out_lstm.shape[-1]))
        return out_linear

    def init_state(self, batch_size):
        h = torch.zeros(size=(1, batch_size, self.hidden_size))
        c = torch.zeros(size=(1, batch_size, self.hidden_size))
        return h, c


# 实例化网络模型 ==================================================
lstm = LstmModel(vocab_size=vocab_size,
                      embed_size=embed_size,
                      hidden_size=hidden_size,
                      output_size=output_size)

# 定义损失函数 ====================================================
loss_fun = nn.CrossEntropyLoss()

# 定义优化器 ======================================================
lr = 0.001
optimizer = torch.optim.Adam(LstmModel.parameters(), lr=lr)

#################################################################
# 定义参数，训练网络
#################################################################
epochs = 200
# LstmModel.train()
print('---------------------- 开始训练模型----------------------')
for epoch in range(epochs):
    loss_all = 0
    for x, y in data_iter:
        optimizer.zero_grad()
        y_pre = lstm(x)
        loss = loss_fun(y_pre, y.reshape(y.shape[0] * y.shape[1]))
        loss_all = loss_all + loss
        loss.backward()
        optimizer.step()

    print("epoch", epoch, "/", epochs, "total loss is", loss_all)

torch.save(lstm.state_dict(), "model.pth")

#################################################################
# 测试一下训练效果
#################################################################
text_test = ["海", "贼", "王", "的", "主", "角", "是"]
list_test = []
len_xy_sub = len(text_test)
for j in range(88):
    if j < len_xy_sub:
        list_test.append(dit_x[text_test[j]])
    # 按照最长样本序列对较短序列进行填充 -----------------------------
    else:
        list_test.append(dit_x["PAD"])

# 计算测试文本的预测值 ---------------------------------------------
tensor_test = torch.tensor(np.array(list_test))
tensor_test = tensor_test.reshape(1,88)
pre_test = LstmModel(tensor_test)
pre_test_id = torch.argmax(pre_test, dim=1)

# 将标签的字符和数值对换，生成一个新的字典 -----------------------------
dit_y_rev = {}
for k in dit_y:
    dit_y_rev[dit_y[k]] = k
pre_test_label = [dit_y_rev[y] for y in pre_test_id.tolist()][0:len_xy_sub]
print(text_test)
print(pre_test_label)