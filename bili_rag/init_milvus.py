# 向 Milvus 中插入数据
from pymilvus import MilvusClient
from embedding import EmbeddingModel
from tqdm import tqdm
import numpy as np

def init_milvus(file, chunk_size=50):
    """
    初始化 milvus, 向其中插入数据
    chunk_size: 模型每次处理多少行数据,取决于显存和内存大小
    """
    # 初始化milvus数据库
    milvus_client = MilvusClient(uri="./data/milvus.db")
    collection_name = "bili_comments"

    # 检查 Collections 是否已存在，如果已存在，则将其删除
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    # 使用指定参数创建新 Collections
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=768, # iic/nlp_gte_sentence-embedding_chinese-base 的默认向量维度
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
    )

    # 构造 Embedding model
    embedding_model = EmbeddingModel()

    # 读数据集，插入数据
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    tasks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    for i, t in enumerate(tqdm(tasks)):
        # 每处理完一个批次就异步插入数据
        result = embedding_model.get_embedding(t)
        insert_data = []
        for j, x in enumerate(result["text_embedding"]):
            insert_data.append({"id": i*chunk_size+j, "vector": x.tolist(), "text": t[j]})
        milvus_client.insert(collection_name=collection_name, data=insert_data)
    
    milvus_client.close()

if __name__ == '__main__':
    init_milvus("./data/comments.txt")