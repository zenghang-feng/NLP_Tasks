import numpy as np

############################################################
# 读取数据，预处理数据
############################################################
# 读取数据 ==================================================
file_name = "文本截取.txt"
list_text = []
with open(file=file_name, mode="r", encoding="utf8") as f:
    text = f.readline()
    while text:
        text = text.replace("。/wj", "").replace("，/wd", "").replace("、/wu", "").replace("\n", "")
        l_text = text.split(" ")
        l_text = [t for t in l_text if t != ""]
        list_text.append(l_text)
        text = f.readline()

# 提取数据中的字符和词性标签，转换数据格式 ========================
list_text_pos, list_text_w = [], []
dit_pos, dit_w = {}, {}
i1, i2 = 0, 0
for text in list_text:
    tmp_pos, tmp_w = [], []
    for wp in text:
        tmp = wp.split("/")

        pos = tmp[1]
        if pos not in dit_pos:
            dit_pos[pos] = i1
            i1 += 1
        tmp_pos.append(pos)

        w = tmp[0]
        if w not  in dit_w:
            dit_w[w] = i2
            i2 += 1
        tmp_w.append(w)

    list_text_pos.append(tmp_pos)
    list_text_w.append(tmp_w)


############################################################
# 计算转移矩阵，发射矩阵
############################################################
# 计算频数 ==================================================
len_pos = len(dit_pos)
dit_pos["pi"] = len_pos
len_w = len(dit_w)

mat_trans = np.zeros(shape=(len_pos+1, len_pos))
mat_emission = np.zeros(shape=(len_pos, len_w))

len_list_pos= len(list_text_pos)
for i in range(len_list_pos):
    list_pos_sub = list_text_pos[i]
    len_list_text_sub = len(list_pos_sub)
    pos_s = "pi"
    list_w_sub = list_text_w[i]
    for j in range(len_list_text_sub):
        # 填充转移矩阵 --------------------------------------
        pos_e = list_pos_sub[j]
        mat_trans[dit_pos[pos_s], dit_pos[pos_e]] += 1
        pos_s = pos_e

        # 填充发射矩阵 --------------------------------------
        idx_r = list_pos_sub[j]
        idx_c = list_w_sub[j]
        mat_emission[dit_pos[idx_r], dit_w[idx_c]] += 1

# 计算概率，并进行平滑处理 ====================================
epslo = 0.0001
mat_trans = mat_trans + epslo
mat_emission = mat_emission + epslo
# 计算转移矩阵概率 -------------------------------------------
r_t, c_t = mat_trans.shape
for i in range(r_t):
    row_sum = np.sum(mat_trans[i,:])
    for j in range(c_t):
        mat_trans[i,j] = mat_trans[i,j] / row_sum
# 计算发射矩阵概率 -------------------------------------------
r_e, c_e = mat_emission.shape
for j in range(c_e):
    col_sum = np.sum(mat_emission[:,j])
    for i in range(r_e):
        mat_emission[i,j] = mat_emission[i,j] / col_sum

############################################################
# 维特比算法
############################################################
# 输入是至少包含2个单词的字符串 ==============================
text_new = "中国 的 经济 进一步 改善"
# text_new = "中国 和 美国 的 合作 不断 扩大"
new_list = text_new.split(" ")
mat_c = np.zeros(shape=(len_pos, len(new_list)))
mat_d = np.zeros(shape=(len_pos, len(new_list)))

# 初始化矩阵C、D，填充第1列 ==================================
for i in range(len_pos):
    mat_c[i, 0] = mat_trans[len_pos, i] * mat_emission[i, dit_w[new_list[0]]]
    mat_d[i, 0] = 0

# 前向计算概率 ==============================================
for j in range(1, len(new_list)):
    for i in range(len_pos):
        prb_max = 0
        # prb_max = - np.inf
        i_b_max = 0
        for i_b in range(len_pos):
            prb_tmp = mat_c[i_b, j-1] * mat_trans[i_b, i] * mat_emission[i, dit_w[new_list[j]]]
            # prb_tmp = np.log(mat_c[i_b, j-1]) + np.log(mat_trans[i_b, i]) + np.log(mat_emission[i, dit_w[new_list[j]]])
            if prb_max < prb_tmp:
                prb_max = prb_tmp
                i_b_max = i_b
        # 填充最大概率和最大概率对应的词性索引 ----------------
        mat_c[i,j] = prb_max       
        mat_d[i,j] = i_b_max


# 后向计算提取词性标签 =======================================
dit_pos_rev = {}
for k in dit_pos:
    dit_pos_rev[dit_pos[k]] = k

# 最后一列概率最大对应的词性的行 ==============================
res = []
for j in range(len(new_list)-1, -1, -1):
    if j == len(new_list)-1:
        idx_max = int(np.argmax(mat_c[:,j]))
        
    else:
        idx_max = int(mat_d[idx_max,j+1])
        
    res = [dit_pos_rev[idx_max]] + res


print(text_new)
print(res)