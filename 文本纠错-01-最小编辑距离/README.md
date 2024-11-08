# 1.项目概述

本文采用最小编辑距离进行文本纠错，这里的文本纠错指的是英文单词或者汉语拼音等拼写的错误；


# 2.程序实现

```
import numpy as np

#################################################################
# 读取英文文本数据
#################################################################
file_name = "ctext.txt"
with open(file=file_name, mode="r", encoding="utf8") as f:
    str_text = f.readlines()[0]

# 去除标点符号，并进行大小写转换 =====================================
str_text = str_text.replace(",", "").replace(".", "")
str_text = str_text.lower()
list_strs = str_text.split(" ")

# 生成词典，并进行词的出现次数统计 ===================================
dit_w = {}
dit_w_c = {}
for w in list_strs:
    if w not in dit_w:
        dit_w[w] = 1
        dit_w_c[w] = 1
    else:
        dit_w_c[w] += 1

#################################################################
# 最小编辑距离函数
#################################################################
def min_edit_dis(str1: str, str2: str) -> int:
    str1 = "#" + str1
    str2 = "#" + str2
    len1 = len(str1)
    len2 = len(str2)
    # 初始化计数矩阵 ------------------------
    arr_dis = np.zeros(shape=(len1, len2))
    for i in range(len1):  # 矩阵第一列初始化
        arr_dis[i, 0] = i
    for j in range(len2):  # 矩阵第一行初始化
        arr_dis[0, j] = j
    # 动态规划计算最小编辑距离 ----------------
    for i in range(1, len1):
        for j in range(1, len2):
            del_dis = arr_dis[i - 1, j] + 1  # 删除成本为1
            ins_dis = arr_dis[i, j - 1] + 1  # 插入成本为1
            if str1[i] == str2[j]:
                rep_dis = arr_dis[i - 1, j - 1]
            else:
                rep_dis = arr_dis[i - 1, j - 1] + 2  # 替换成本为2
            arr_dis[i, j] = min(del_dis, ins_dis, rep_dis)

    return int(arr_dis[i, j])

#################################################################
# 通过筛选具有最小编辑距离的词的词频，取词频最高的词
#################################################################
str_ori = "sprnd"
# 遍历词典，计算词典中与目标词编辑距离最小的词 ======================
dit_len_w = {}
len_min = len(str_ori) * 2
for w in dit_w:
    len_tmp = min_edit_dis(str_ori, w)

    if len_tmp in dit_len_w:
        dit_len_w[len_tmp] = dit_len_w[len_tmp] + " " + w

    if len_tmp not in dit_len_w:
        dit_len_w[len_tmp] = w

    if len_tmp < len_min:
        len_min = len_tmp

list_w_cond = dit_len_w[len_min].split(" ")

# 计算编辑距离最小的全部词出现的概率，取出现概率最大的词作为纠正 =======
prob = 0
w_res = ""
for w in list_w_cond:
    prob_tmp = dit_w_c[w] / len(list_strs)
    if prob_tmp > prob:
        prob = prob_tmp
        w_res = w
print("输入的单词：", str_ori, ";", "纠正后的单词", w_res)
```
