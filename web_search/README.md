# Bochaai RAG (OLLAMA + Milvus)
## 主要功能
1. **Web 检索 API**：接入博查AI开放平台API，基于多模态混合搜索和语义排序技术的新一代搜索引擎，实现商业化自然语言搜索
2. **LLM 能力**：LLM 部署在 Ollama 上，实现跨平台和跨设备的简单调用
3. **数据存储**：Milvus 作为数据存储，实现海量数据的存储和检索
4. **RAG WorkFlow**：基于 LangGraph 实现，可方便地调取工具和保存历史记录
5. **检索方式**：实现本地数据库+在线检索的混合检索方式，提供实时更新本地数据库的功能

## Instruction
1. **LLM 后台配置**
- 确认 Ollama 环境正常，安装指南见 [marshwallen/llm-deploy-playground](https://github.com/marshwallen/llm-deploy-playground)
- 这里的 Ollama URL 以 ```http://localhost:11434``` 为例，LLM 以 ```deepseek-r1:8b``` 为例
```sh
ollama -v
# ollama version is 0.5.7

ollama list
# NAME               ID              SIZE      MODIFIED
# deepseek-r1:14b    ea35dfe18182    9.0 GB    3 days ago
# deepseek-r1:8b     28f8fd6cdc67    4.9 GB    3 days ago
```

2. **申请 Bochaai 开放平台服务**
- 官方链接：https://open.bochaai.com/overview
- 在 API KEY 管理页申请 API，在 ```/config.yaml``` 文件中配置
```sh
bochaai:
  url: https://api.bochaai.com/v1/web-search
  api_key: sk-xxx
```

3. **运行 RAG 服务**
- 运行以下命令即可，开启多轮聊天
- 键入 ```exit``` 退出
```sh
python run.py --ollama_url http://localhost:11434 --ollama_model deepseek-r1:8b
```

## WorkFlow 介绍
### 此 RAG 采用分层检索架构设计
1. 在进行检索时，会先去本地的 Milvus 服务器/数据库文件检索相关内容，随后进行相关性评估
2. 评估后会根据相关性的不同去决定是否调用 API 进行在线检索
3. 在线检索后，会将结果合并到本地数据库中，方便下次快速查找，同时减少对公共互联网资源的负担
4. 相关性评估使用 ```cross-encoder/ms-marco-MiniLM-L-6-v2``` 交叉编码器，通过比对 query 和检索内容，能够显著提升信息检索（IR）系统的质量
5. 通过正则化手段匹配 query 与检索内容的一些关键词，减少一些相似但无关的内容
6. 最后的检索结果是本地数据库资源和在线检索结果的结合

### 一图流如下
```sh
A[用户提问] --> B{本地检索}
B -->|结果存在| C[相关性评估]
B -->|无结果| D[网络检索]
C -->|相关| E[生成回答]
C -->|不相关| D
D --> F[合并结果]
F --> E
```

# Reference
- https://aq6ky2b8nql.feishu.cn/wiki/HmtOw1z6vik14Fkdu5uc9VaInBb
- https://langchain-ai.github.io/langgraph/#example
- https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2

