# RAG-Milvus
from glob import glob
from openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from embedding import emb_text
from utils import config
from init_milvus import init_milvus
import os
from pymilvus import MilvusClient

class MilvusRAG:
    def __init__(self):
        self.cfg = config()
        # 使用的平台名字
        self.p_name = self.cfg["used"]

        # 初始化向量数据库milvus
        db_file = "./data/milvus_demo.db"
        if not os.path.exists(db_file):
            init_milvus()
        
        self.milvus_client = MilvusClient(uri=db_file)
        self.collection_name = "rag_c_0"  

        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.cfg["platform"][self.p_name]["embedding_dim"],
            metric_type="IP",
            consistency_level="Strong",
        ) 

        # 加载模型
        self.client = OpenAI(
            api_key=self.cfg["platform"][self.p_name]["api_key"], 
            base_url=self.cfg["platform"][self.p_name]["base_url"],
            )
        
    def run(self, question):
        # 1 先在数据库中检索：在 Collections 中搜索该问题并检索语义前 3 个匹配项
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[
                emb_text(self.p_name, question)
            ],
            limit=3,  # Return top 3 results
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],  # Return the text field
        )
        # distance 为相似度
        retrieved_lines_with_distances = [
            (res["entity"]["text"], res["distance"]) for res in search_res[0]
        ]

        # 检索到的文档
        context = "\n".join(
            [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
        )

        # 定义系统和用户提示
        SYSTEM_PROMPT = """
        Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided. If the following prompt is empty, then you should return an empty answer.
        """
        USER_PROMPT = f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """

        # 调用 API
        completion = self.client.chat.completions.create(
            model=self.cfg["platform"][self.p_name]["model"],
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': USER_PROMPT}],
            )
        
        ans = completion.choices[0].message.content
        print(f"{'#'*10}\n{ans}\n{'#'*10}\n")

if __name__ == "__main__":
    rag = MilvusRAG()
    q = "How is data stored in milvus?"
    rag.run(q)




