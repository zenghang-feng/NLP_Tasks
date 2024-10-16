from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatZhipuAI
from langchain.chains.retrieval_qa.base import RetrievalQA


######################################################################
# 引入预训练的embedding模型
######################################################################
ak = "<YOUR_OPENAI_API_KEY>"            # 智谱Ai的密钥
zp_embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=ak)


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

print("-----------")