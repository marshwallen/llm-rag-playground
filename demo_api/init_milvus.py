# 将数据载入 Milvus
from pymilvus import MilvusClient
from tqdm import tqdm
from utils import config
from glob import glob
from embedding import emb_text

def init_milvus():
    cfg = config()
    p_name = cfg["used"]

    # 初始化milvus数据库
    milvus_client = MilvusClient(uri="./data/milvus_demo.db")
    collection_name = "rag_c_0"

    # 检查 Collections 是否已存在，如果已存在，则将其删除
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    # 使用指定参数创建新 Collections
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=cfg["platform"][cfg["used"]]["embedding_dim"],
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
    )

    # 准备数据：使用Milvus 文档 2.4.x中的常见问题页面作为 RAG 中的私有知识，
    # 这对于简单的 RAG 管道来说是一个很好的数据源
    text_lines = []
    for file_path in glob("data/milvus_docs/en/faq/*.md", recursive=True):
        with open(file_path, "r") as file:
            file_text = file.read()

        # 仅用"#"分割文件中的内容
        text_lines += file_text.split("# ")

    # 注意这里插入的数据形式
    # 文本 -> 嵌入向量 -> Milvus 文档
    data = []
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": emb_text(p_name, line), "text": line})

    milvus_client.insert(collection_name=collection_name, data=data)
    milvus_client.close()
    