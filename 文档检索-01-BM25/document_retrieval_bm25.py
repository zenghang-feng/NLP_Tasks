import re
import jieba
import pandas as pd
import numpy as np

##############################################################################
# 数据准备
##############################################################################
# 读取停用词表 =================================================================
file_stop_words = "hit_stopwords.txt"
with open(file_stop_words, 'r', encoding='utf-8') as f:
    stop_w_l = [word.strip('\n') for word in f.readlines()]

# 读取和处理知识库文本 ==========================================================
file_klg = "knowledge_docs.xlsx"
df_klg = pd.read_excel(file_klg)
# 对文档分词，生成的结果存储在列表中，每个词作为列表中一个元素 -------------------------
df_klg['文档内容_分词'] = df_klg.apply(lambda row: jieba.lcut(re.sub(r'[\W]', '', row['文档内容'])), axis=1)
# 去除停用词，生成的结果存储在字符串中，每个词之间以空格分隔 --------------------------
df_klg['文档内容_分词_去停用词'] = df_klg.apply(lambda row: [w for w in row['文档内容_分词'] if w not in stop_w_l], axis=1)
# 将文档的分词列表存储在一个列表中 -----------------------------------------------
docs_words = df_klg['文档内容_分词_去停用词'].tolist()

# 读取和处理用户问题 ============================================================
query_str = "中国和美国的经济哪个好"
query_words = [w for w in jieba.lcut(re.sub(r'[\W]', '', query_str)) if w not in stop_w_l]


##############################################################################
# 实现BM25算法
##############################################################################
def bm25(query: list, docs: list, k1: float, b: float) -> list:
    # 计算各个文档的平均长度 --------------------------------
    doc_nums = len(docs)
    avgdl = sum([len(doc) for doc in docs]) / doc_nums

    # 计算各个文档中的词的词频 ------------------------------
    docs_wc = []
    for doc in docs:
        doc_words_count = {}
        for w in doc:
            if w in doc_words_count:
                doc_words_count[w] += 1
            else:
                doc_words_count[w] = 1
        docs_wc.append(doc_words_count)

    # 计算query中每个词所出现在的文档的数量 ------------------
    query_wds = {}
    for wq in query:
        count = 0
        for doc_words_count in docs_wc:
            if wq in doc_words_count:
                count += 1
        query_wds[wq] = count

    # 计算query与每个文档的相似度分值 -----------------------
    res = []
    for i in range(doc_nums):
        doc = docs[i]
        doc_words_count = docs_wc[i]
        score_sum = 0
        for wq in query:
            n_wq = query_wds[wq]
            idf_wq = np.log((doc_nums - n_wq + 0.5) / (n_wq + 0.5) + 1)

            f_wq = 0
            if wq in doc_words_count:
                f_wq = doc_words_count[wq]
            score = idf_wq * (f_wq * (k1 + 1)) / (f_wq + k1 * (1 - b + b * (len(doc)/avgdl)))
            score_sum = score_sum + score

        res.append(score_sum)

    return res


##############################################################################
# 进行文档检索
##############################################################################
# type = isinstance([1, 2], list)
res = bm25(query=query_words, docs=docs_words, k1=1.5, b=0.75)
idx = res.index(max(res))
print(query_words)
print(docs_words[idx])
