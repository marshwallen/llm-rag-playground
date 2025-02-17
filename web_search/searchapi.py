# RAG 搜索 API
# 以博查世界知识搜索引擎为例
# 开放平台：https://open.bochaai.com/overview
# URL: https://api.bochaai.com/v1/web-search

import requests
import json
import os
from pymilvus import MilvusClient
from embedding import EmbeddingModel
import hashlib
import yaml

class BochaaiSearchAPI:
    def __init__(self):
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.url = cfg["bochaai"]["url"]
        self.api_key = cfg["bochaai"]["api_key"]
        self.name = "bochaai"

        # 初始化向量数据库milvus
        os.makedirs("data/", exist_ok=True)
        self.db = "data/bochaai_milvus.db"
        self.milvus_client = MilvusClient(uri=self.db)
        self.collection_name = self.name
        
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=768,
            metric_type="IP",
            consistency_level="Strong",
        ) 

        # 初始化 Embedding 模型
        self.embedding = EmbeddingModel()

    def _request(self, payload:dict):
        """HTTP 请求
        Websearch 请求体文档：https://aq6ky2b8nql.feishu.cn/wiki/RXEOw02rFiwzGSkd9mUcqoeAnNK
        """
        payload = json.dumps(payload)
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        try:
            response = requests.request("POST", self.url, headers=headers, data=payload)
        except Exception as e:
            print(f"Request Error {e}")
        try:
            response = response.json()["data"]["webPages"]["value"]
        except Exception as e:
            print(f"Request Error {e}, response: {response}")

        return response
    
    def _insert_data(self, data:list, dtype, chunk_size=50):
        """向 Milvus 中分批插入数据
        dtype: 需要转换为 Vector 的字段
        """
        tasks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        for _, t in enumerate(tasks):
            # 每处理完一个批次就异步插入数据
            vector = self.embedding.get_embedding([x[dtype] for x in t])
            insert_data = []
            for j in range(len(vector["text_embedding"])):
                content = {
                        "vector": vector["text_embedding"][j].tolist(), 
                    }
                content.update(t[j])
                content.update({"id": self._generate_decimal_id(t[j]["name"])})

                insert_data.append(content)

            self.milvus_client.upsert(collection_name=self.collection_name, data=insert_data)

    def _generate_decimal_id(self, text: str) -> int:
        """根据 URL Name 生成固定长度的十进制10位数字ID
        """
        # 生成哈希并转为大整数
        hash_bytes = hashlib.sha256(text.encode()).digest()
        hash_num = int.from_bytes(hash_bytes, byteorder='big')
        
        return hash_num % (10**10)
    
    def _search_milvus(self, query, limit, dtype) -> list[str]:
        """
        到 milvus 数据库中检索内容
        直接返回查到的 text 内容
        """
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=self.embedding.get_embedding([query])["text_embedding"].tolist(),
            limit=limit,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=[dtype],
        )
        context = [x["entity"][dtype] for x in search_res[0]]
        return context

    def search(self, query, local:bool, limit=10, dtype="summary") -> list[str]:
        """
        从数据库/在线内容请求数据
        local：布尔值，是否在本地或在线内容中寻找数据
        dtype: Web查询时返回的字段类型（summary or snippet）
        """
        # 找本地数据
        if local:
            res = self._search_milvus(query, limit, dtype)
            print("# Using Local Search")
        else:
        # 本地数据解决不了请求，转而通过 search api 在线访问网页，同时将抓取到的内容存入数据库中
            res = self._request({
                    "query": query,
                    "freshness": "noLimit",
                    "summary": True,
                    "count": limit
                })
            self._insert_data(res, dtype)
            res = [x[dtype] for x in res]
            print("# Using Online Web Search")

        return res
                
    def close(self):
        """
        关闭 Milvus 客户端
        """
        self.milvus_client.close()
