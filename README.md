# RAG+Milvus
本 repo 是 RAG+Milvus 的应用实践
- Milvus+RAG: https://milvus.io/docs/zh/build-rag-with-milvus.md

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

- **LLM**: Qwen (API)
- **Embedding model**: Multimodal-embedding-v1 (API)
- **Database**: Milvus
- **Data source**: Milvus Docs (milvus_docs_2.4.x_en)

1. **下载并解压数据**
```sh
# Download milvus_docs_2.4.x_en.zip
cd simple_rag/
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

## Bilibili RAG
通过 LLM 与本地 Milvus 的交互，实现 Bilibili 某些 UP主 下视频评论区内容的检索。检索构造的 SYSTEM_PROMPT 如下：
```python
# 特殊符号定义
unknow_symbol = "<unk>"
answer_symbol = "<ans>"
tips_symbol_s = "<tip>"
tips_symbol_e = "</tip>"
max_tries = 5

system_prompt = f"""
    人类：你是一个人工智能助手，我会问你一些问题。首先你应该不依赖外部检索尝试给出问题的答案，如果不确定问题的答案，请返回 {unknow_symbol} 以提示本地去检索。如果答案确定，请返回 {answer_symbol} 以结束问答。
    你只有 {max_tries} 次机会尝试回答这个问题。每次结束回答且无法给出问题的答案时，你可以用 {tips_symbol_s} 来告诉我你不知道的部分，以提示我继续尝试本地检索。例如，你可以用 {tips_symbol_s} XXX {tips_symbol_e} 来包裹问题，其中XXX是你想进一步询问我的问题。记住，无论在什么情况下，你都需要返回之前所定义的三种特殊符号以结束问答。
    """
```

- **LLM**: Qwen (API)
- **Embedding model**: nlp_gte_sentence-embedding_chinese-base (Local)
- **Database**: Milvus
- **Data source**: Bilibili Comments

1. **获取B站评论区数据**
获取方法移步至下方repo

- Github: https://github.com/NanmiCoder/MediaCrawler
- 本 repo 使用的部分配置如下：
```python
# 文件目录：MediaCrawler/config/base_config.py
# 爬取一级评论的数量控制(单视频/帖子)
CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = 20

# 是否开启爬二级评论模式, 默认不开启爬二级评论
# 老版本项目使用了 db, 则需参考 schema/tables.sql line 287 增加表字段
ENABLE_GET_SUB_COMMENTS = True

# 指定bili创作者ID列表(sec_id)
# 极客湾 + 笔吧测评室
BILI_CREATOR_ID_LIST = [
    "25876945",
    "367877"
]
```
- 执行命令
```sh
python main.py --platform bili --lt qrcode --type creator
```
- 数据拷贝
```MediaCrawler/data/bilibili/json/```目录下的所有json文件拷贝至```rag-milvus/bili_rag/data/```即可

2. **处理获取到的 Bilibili 评论数据**
主要是将结构化的 json 数据转换成一个个句子存放到 txt 文件中备用：
```sh
# 每个句子的形式如下
# s = "[ID: xxx][视频发布时间：xxxx年xx月xx日][视频标题：xxx][视频简介：xxx][评论时间：xxxx年xx月xx日][评论者昵称：xxx][父级评论：xxx][评论内容：xxx]"

# 执行命令
cd bili_rag/
python prepare_data.py

# [可选]手动向 milvus 数据库中插入上述 embedding 后的数据（初始化）
python init_milvus.py
```

3. **运行 Demo**
```sh
# 配置文件在 bili_rag/config.yaml，需要获得对应API的Key
# question 在 main 中调整
python main.py
```
### Bilibili RAG 测试输出效果
注：这里隐去了用户昵称（nickname）
```sh
# 尝试次数: 1
# user_prompt: B站用户对骁龙888的评价如何？
# tips: 
# context: 
# answer: <tip> 我没有直接的数据或评论来源关于B站用户对骁龙888的具体评价。你是否可以提供一些更具体的信息或者例子？ </tip>
/home/marshwallen/anaconda3/envs/basellm/lib/python3.12/site-packages/transformers/modeling_utils.py:1044: FutureWarning: The `device` argument is deprecated and will be removed in v5 of Transformers.
  warnings.warn(
# 尝试次数: 2
# user_prompt: B站用户对骁龙888的评价如何？
# tips:  我没有直接的数据或评论来源关于B站用户对骁龙888的具体评价。你是否可以提供一些更具体的信息或者例子？ 
# context: 还有，骁龙820和888的往事还历历在目，说不定差距不会越拉越大，双向奔赴的可能性也不小]
骁龙888→1514
骁龙888→1514
高通888是因为三星代工太烂了，所以才翻车，高通888在设计上没有任何问题，如果888采用台积电代工，完全能超越麒麟9000[吃瓜]
跑分大于888小于8+
8gen1在实际性能上也和888差不太多，所以基本还是888和8gen1这个范畴的。
还有，骁龙820和888的往事还历历在目，说不定差距不会越拉越大，双向奔赴的可能性也不小][评论内容: 回复 @xxx :mtk在啊谁说没竞争]
其他友商的旗舰机 就差B站评论区那些懂王人手一部了。 要不然高端在高通 865的时候就冲上去了[笑哭]]
骁龙810：火龙[热]][评论内容: 没有经历过820/821导致的，其实888比820/821及之前的处理器能效好多了，不过比起865一代还倒退不少]
如果麒麟9000s真是888同款性能，这个表现很夸张了][评论内容: 回复 @xxx :是真的，我已经用上了，如果你什么都没感受到，但你不可能不感受到他的流畅性。可以说其它都是小弟]

# answer: <ans>根据提供的信息片段，B站用户对骁龙888的评价褒贬不一。有用户认为高通888的设计没有问题，其性能表现不佳是因为三星代工的问题，甚至有评论指出如果888采用台积电代工的话，完全可以超越麒麟9000。另有评论表示，888相较于820/821在能效上有所提升，但与865相比仍有所倒退。还有用户提到，其他厂商的旗舰机在高端市场上未能迅速崛起，可能也暗示了对包括骁龙888在内的处理器性能的一种期待或不满。</ans>
```


