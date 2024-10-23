import re
import jieba
import jieba.analyse

#################################################################
# 读取文本数据
#################################################################
file_name = "text_content.txt"
with open(file=file_name, mode="r", encoding="utf8") as f:
    text = f.read()                     # 读取文本中全部内容
    text = text.replace('\n', '')       # 去掉文本中的换行符

#################################################################
# classifier4j进行提取式文本摘要
#################################################################
# 获取文本中k个高频词/关键词 ========================================
# 分词，去停用词，获取关键词 -----------------------------------------
k = 4                                                                       # 需要适当调整超参数
key_w_list = []
for w, p in jieba.analyse.textrank(text, topK=k, withWeight=True):
    key_w_list.append(w)

# 将文本数据拆分成对应的句子 ========================================
sts = re.split(pattern="。|！|\!|\.|？|\?", string=text)                     # 分割时去掉句子间分隔符

# 获取包含关键词的前n个句子 =========================================
res_s = []
for s in sts:
    flag = 'y'
    for w in key_w_list:
        if s.find(w) == -1:
            flag = 'n'
            break                                                           # 跳出本层循环
    if flag == 'y':
        res_s.append(s)

# 将句子按照在文本中出现的顺序排序，添加适当分隔符后输出 ==================
# 上文中句子是按照顺序排列的，所以直接输出 -----------------------------
text_res = "。".join(res_s)
print("--------------------摘要提取完成--------------------------")