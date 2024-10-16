from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma


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
print("-----------")