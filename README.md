# LLM-RAG-Playground
本 repo 是 RAG 的应用实践。
参考：https://www.elastic.co/search-labs/blog/retrieval-augmented-generation-rag

![image](https://www.elastic.co/search-labs/_next/image?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fme0ej585%2Fsearch-labs-import-testing%2F041725d7399658012a41719e1660072fb2b2e608-1260x725.png&w=1920&q=75)

# 各平台API
- OpenAI: https://platform.openai.com/docs/api-reference/introduction
- 通义千问: https://bailian.console.aliyun.com/#/model-market
- DeepSeek V3: https://api-docs.deepseek.com/zh-cn/

# Instruction
- 安装环境依赖
```sh
pip install -r requirements.txt
```

## Simple RAG
构建简单 RAG Demo，实现对 Milvus 官方文档的检索
- https://github.com/marshwallen/llm-rag-playground/tree/main/simple_rag

## Bilibili RAG (Milvus)
通过 LLM 与 Milvus 数据库的交互，实现 Bilibili 某些 UP主 下视频评论区内容的检索
- https://github.com/marshwallen/llm-rag-playground/tree/main/bili_rag_milvus

## Bilibili RAG (Elasticsearch)
通过 LLM 与开源 Elasticsearch 数据库的交互，实现 Bilibili 某些 UP主 下视频评论区内容的检索
- https://github.com/marshwallen/llm-rag-playground/tree/main/bili_rag_es


