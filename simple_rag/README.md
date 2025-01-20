## Simple RAG
构建简单 RAG Demo，实现对 Milvus 官方文档的检索

- **LLM**: Qwen (API)
- **Embedding model**: Multimodal-embedding-v1 (API)
- **Database**: Milvus
- **Data source**: Milvus Docs (milvus_docs_2.4.x_en)

## Instruction

1. **下载并解压数据**
```sh
# Download milvus_docs_2.4.x_en.zip
sh prepare_data.sh

# [可选]手动向 milvus 数据库中插入上述 embedding 后的数据（初始化）
python init_milvus.py
```

2. **运行 Demo**
```python
# 配置文件在 simple_rag/config.yaml，需要获得对应API的Key
# question 在 main 中调整
python main.py
```