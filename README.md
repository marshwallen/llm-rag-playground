# RAG+Milvus Simple Demo
- Milvus+RAG: https://milvus.io/docs/zh/build-rag-with-milvus.md

# 各平台API
- OpenAI: https://platform.openai.com/docs/api-reference/introduction
- 通义千问: https://bailian.console.aliyun.com/#/model-market
- DeepSeek V3: https://api-docs.deepseek.com/zh-cn/

# Instruction
1. 准备数据集
```sh
cd RAG-Milvus
sh prepare_data.sh
```

2. 运行 Demo
```python
python main.py
```
输入模型的question在主函数中调整
