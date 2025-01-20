# Elasticsearch 客户端
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import os

class ESClient:
    def __init__(self, hosts="http://localhost:9200"):
        """
        创建一个 ES 客户端
        """
        # 如果检测到 ES 容器不存在，则启动容器
        if not os.path.exists("elastic-start-local/.env"):
            raise Exception("Please run start-local.sh first.")
        
        # 读取 .env 文件，写入字典
        self._env = {}
        with open("elastic-start-local/.env", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                self._env[key] = value

        api_key = self._env["ES_LOCAL_API_KEY"]
        self._client = Elasticsearch(
            hosts=hosts, 
            api_key=api_key
            )
        
        print("Connected to Elasticsearch!")
    
    def insert(self, index, body: dict):
        """
        批量插入数据。index 是表名, doc_id为文档唯一标识符, body是插入体
        默认覆盖相同doc_id的文档
        body格式: {id_0: {...}, id_1: {...}, ...}
        """
        documents = [
            {
                "_index": index,
                "_id": k,
                "_source": v
            } for k,v in body.items()
        ]
        bulk(self._client, documents)

    def search(self, index, body):
        """
        查找数据
        """
        response = self._client.search(index=index, body=body)
        return response

    def close(self):
        """
        关闭 Elasticsearch 客户端
        """
        self._client.close()