# 1. 系统实现方式

**RAG(检索增强生成)** ： 可以整合自定义知识库中的信息，并以这些信息为基础，输入给LLM生成较为精准的答案。

整体结构如下图所示：

![i](https://github.com/zenghang-feng/NLP_Tasks/blob/main/问答系统-01_RAG/图片附件/01-RAG架构.jpg)

## 1.1 **数据读取** ：

第一步是读取用于构建知识库的文本数据，如果单个文本过长，需要对文本进行分割，以满足模型输入的要求。  

下文程序中，知识库各个文本存储在一个PDF文档中，每个文本以MarkDown的一级标题标识。  

程序首先读取PDF文档，将所有Page的内容拼接，然后通过MarkDown标题对文本进行切分。  

## 1.2 **向量嵌入** ：  

第二步，通过预训练的Embedding模型，将1.1中处理后的文档进行向量化，并持久化存储到向量数据库中。  

通过Embedding将文本进行向量化可以参考下图理解：  

![i](https://github.com/zenghang-feng/NLP_Tasks/blob/main/问答系统-01_RAG/图片附件/02-Embedding-1.jpg)

![i](https://github.com/zenghang-feng/NLP_Tasks/blob/main/问答系统-01_RAG/图片附件/02-Embedding-2.jpg)

下文程序中采用的是智谱AI的预训练Embedding模型，向量数据库采用的是Chorma。  

注意：从这一步开始，需要准备一个智谱AI的API_KEY，直接从官网注册就可以（也可以采用ChatGpt、文心一言等模型）。  

## 1.3 **向量检索** ：  

第三步，根据用户输入的问题文本，用Embedding模型将用户问题文本嵌入到与1.2中向量数据库中额外上下文相同的向量空间。  

然后选择相似性检索/MMR检索等不同检索方式，从向量数据库中检索出最接近用户问题的前k个知识库文本。  

下文程序中采用的是余弦相似度检索。

## 1.4 **增强生成** ：  

第四步，首先接入LLM，用于生成问题答案。  

Langchain将用户问题和检索到的知识库文本放入一个提示模板中，输入给接入的LLM模型，以生成对应的答案。  

下文程序中，采用的是智谱AI预训练LLM。

向量检索和答案生成的结构框图如下所示：  

![i](https://github.com/zenghang-feng/NLP_Tasks/blob/main/问答系统-01_RAG/图片附件/03-Retrieval.jpg)

# 2. 程序实现

```
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatZhipuAI
from langchain.chains.retrieval_qa.base import RetrievalQA


######################################################################
# 按照markdown文件的标题分割
######################################################################
# 读取所需的文档
loader = PyMuPDFLoader("knowledge_db.pdf")
papers = loader.load()
# 将读取的文档存储在一个字符串中，用于后续进行文本分割
papers_str = ""
for p in papers:
    papers_str = papers_str + p.page_content

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(papers_str)

######################################################################
# 引入预训练的embedding模型
######################################################################
ak = "<YOUR_OPENAI_API_KEY>"            # 智谱Ai的密钥
zp_embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=ak)


######################################################################
# 构建向量数据库，将知识库文档持久化到本地文件
######################################################################
pri_fold = "C:/Users/."                 # 持久化存储的目录地址
knowledge_db = Chroma.from_documents(documents=md_header_splits, embedding=zp_embeddings, persist_directory=pri_fold)
knowledge_db.persist()
print("------知识库文本向量化完成-----")


######################################################################
# 读取向量数据库中的知识
######################################################################
pri_fold = "C:/Users/."                 # 持久化存储的目录地址
knowledge_db = Chroma(persist_directory=pri_fold, embedding_function=zp_embeddings)


######################################################################
# 构建检索问答链
######################################################################
question = "中国和美国的经济前景怎么样？"
zp_llm = ChatZhipuAI(model="glm-4", temperature=0.5, api_key=ak)
# 直接返回生成的答案，没有源文档
chain_qa = RetrievalQA.from_chain_type(llm=zp_llm, retriever=knowledge_db.as_retriever(), chain_type="stuff")
ans = chain_qa.run(question)

# 可以返回源文档
# chain_qa = RetrievalQA.from_chain_type(llm=zp_llm, retriever=knowledge_db.as_retriever(), chain_type="stuff", return_source_documents=True)
# res = chain_qa({"query":question})

print("-----问答完成------")
```
