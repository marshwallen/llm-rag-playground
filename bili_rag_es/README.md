# Bilibili RAG (Elasticsearch)

下图是 ES 官方给出的 RAG 流程。

![image](https://www.elastic.co/search-labs/_next/image?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fme0ej585%2Fsearch-labs-import-testing%2Fc4365c4ce2db46565464495ede89ba6307fecad6-725x746.png&w=1920&q=75)

通过 LLM 与 Elasticsearch 数据库的交互，实现 Bilibili 某些 UP主 下视频评论区内容的检索。

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
- **Database**: Elasticsearch
- **Data source**: Bilibili Comments

## Instruction

1. **获取B站评论区数据**
获取方法移步至下方repo

- https://github.com/NanmiCoder/MediaCrawler
- 本 repo 使用的部分配置如下：
```python
# 文件目录：MediaCrawler/config/base_config.py
# 爬取一级评论的数量控制(单视频/帖子)
CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = 5

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
- 执行命令（需要扫码登陆 Bilibili）
```sh
cd MediaCrawler/
python main.py --platform bili --lt qrcode --type creator
```
- 数据拷贝
```MediaCrawler/data/bilibili/json/```目录下的所有json文件拷贝至当前目录下的```data/```即可。具体的数据格式可见根目录下的```examples/```

2. **安装 Elasticsearch 数据库 (Linux)**
- 安装指南：https://www.elastic.co/guide/en/elasticsearch/reference/8.17/deb.html#deb-repo

```sh
# 官方安装指南
# Download and install the public signing key
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg

# install the apt-transport-https package on Debian before proceeding
sudo apt install apt-transport-https

# Save the repository definition to /etc/apt/sources.list.d/elastic-8.x.list
echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list

# install the Elasticsearch Debian package
sudo apt update && sudo apt install elasticsearch
```

- Elasticsearch 运行在集群上，以及需要 Kibana 来使用 Dev Tools API Console
- 以下命令可以在Docker中建立一个单机本地集群
```sh
# 此命令会在当前目录创建 elastic-start-local/ 文件夹，
# 其中 .env 是基本配置，start.sh 和 stop.sh 分别是启停脚本（支持持久化）, uninstall.sh 卸载
sudo sh start-es-local.sh

# 查看运行情况
sudo docker ps

# 输出
# CONTAINER ID   IMAGE                                                  COMMAND                  CREATED              STATUS                        PORTS                                NAMES
# 5afd5aee0d09   docker.elastic.co/kibana/kibana:8.17.0                 "/bin/tini -- /usr/l…"   About a minute ago   Up About a minute (healthy)   127.0.0.1:5601->5601/tcp             kibana-local-dev
# 5ad4338b9537   docker.elastic.co/elasticsearch/elasticsearch:8.17.0   "/bin/tini -- /usr/l…"   About a minute ago   Up About a minute (healthy)   127.0.0.1:9200->9200/tcp, 9300/tcp   es-local-dev
```

3. **处理获取到的 Bilibili 评论数据**
- 将 json 数据存到 Elasticsearch 中。每个数据条目具有以下内容：

```sh
{
    # 文本
    # "video_create_time:2024-10-14 video_title:xxx video_desc:xxx comment_create_time:2024-10-14 nickname:xxx content:xxx parent_content:xxx"
    "text": {
        "type": "text",
    },
    # 文本对应的 embedding 向量，由 iic/nlp_gte_sentence-embedding_chinese-base 得到
    "vector": {
        "type": "dense_vector",
        "dims": 768
    }
}
```
- 数据初始化执行命令

```python
# 写入 Elasticsearch
python prepare_data.py
```

4. **运行 Demo**
```sh
# 配置文件在 config.yaml，需要获得对应API的Key
# question 在 main 中调整

# ES 数据库的默认匹配算法：余弦相似度
python main.py
```

### Bilibili RAG Elasticsearch 测试输出效果
注：这里隐去了用户昵称（nickname）

```sh
# 尝试次数: 1
# user_prompt: B站用户对骁龙888的评价如何？
# tips: 
# context: 
# answer: <tip> 我没有具体的外部数据或评论内容关于B站用户对骁龙888的评价。您是否可以提供一些更具体的信息或者您可以去B站查看相关视频评论区获取用户的真实反馈？</tip>
/home/marshwallen/anaconda3/envs/basellm/lib/python3.12/site-packages/transformers/modeling_utils.py:1044: FutureWarning: The `device` argument is deprecated and will be removed in v5 of Transformers.
  warnings.warn(
# 尝试次数: 2
# user_prompt: B站用户对骁龙888的评价如何？
# tips:  我没有具体的外部数据或评论内容关于B站用户对骁龙888的评价。您是否可以提供一些更具体的信息或者您可以去B站查看相关视频评论区获取用户的真实反馈？
# context: video_create_time:2024-11-22 video_title:红米K80 Pro上手体验：性能表现不错！ video_desc:我们收到了全新的红米K80 Pro，来看看红米今年的当家花旦表现如何吧！ comment_create_time:2024-11-28 nickname:xxx content:回复 @xxx :我也怀疑，不过下面极客湾有个视频600多w播放测888里还真有原神，只能说这几年过得太快了[喜极而泣] parent_content:我怎么记得原神刚刚发布或者没发布呢 
video_create_time:2024-11-22 video_title:红米K80 Pro上手体验：性能表现不错！ video_desc:我们收到了全新的红米K80 Pro，来看看红米今年的当家花旦表现如何吧！ comment_create_time:2024-11-26 nickname:xxx content:回复 xxx :那也得看用的哪家的888[doge] parent_content:你算准了一切，唯独没算到你朋友用的是888[doge] 
video_create_time:2024-11-22 video_title:红米K80 Pro上手体验：性能表现不错！ video_desc:我们收到了全新的红米K80 Pro，来看看红米今年的当家花旦表现如何吧！ comment_create_time:2024-11-22 nickname:xxx content:首先b站确实降码率，其次一堆人对着静态画面讨论码率，也怪好笑的。 parent_content:None 
video_create_time:2024-12-23 video_title:天玑8400前瞻体验：这个能效有点强！ video_desc:发哥的中端平台终于更新啦！这次的天玑8400居然也用上了全大核设计，搭载8颗A725核心，如此神奇的设计到底能效如何？我们今天就先来前瞻测试一下看看吧…… comment_create_time:2024-12-28 nickname:xxx content:回复 @xxx :正常情况，因为优化很花成本，用户配置高了于是就省成本不优化了。不过这跟 888 不行没啥关系，888 综合表现甚至不如 870，要我说 888 还是换了比较好。 parent_content:回复 @xxx :现在本身手机性能是够用的，最大的问题还是软件供应商的，以前手机性能不好都是堆优化，现在是硬吃手机性能不怎么优化的 
video_create_time:2024-11-02 video_title:高通X Elite深度分析：年度最自信CPU video_desc:普通且自信——这是我用了四个月X Elite笔记本后，对高通这款“划时代”处理器的真实评价。他有一些潜在的闪光点，包括架构设计、省电上的各种创新。但当你在生态和可用性上遥遥落后，性能上又无法隔着转译层碾压各路对手，还在市场定位和价格上痴人说梦时，这款产品就离失败不远了。X Elite到底是什么？为何我会如此评价？看完这期视频你会得到答案…… comment_create_time:2024-11-03 nickname:xxx content:回复 @xxx :888的变种。888是1颗x1超大核，3颗a78大核，4颗a5x小核。8cx gen3是4颗x1，4颗a78 parent_content:8295 不是 888 吗
...
[以下略]
# answer: 从提供的评论片段中可以看出，B站用户对骁龙888的评价褒贬不一，具体如下：

- 有用户提到使用骁龙888设备玩原神能坚持60帧一分钟，暗示在特定场景下的性能尚可。
- 有评论指出经历过888的人大多学会了看测评再做决定，表明这款芯片可能曾经存在一些问题让用户产生了心理阴影。
- 另有用户认为888综合表现甚至不如870，并建议更换更好的处理器。
- 还有观点认为888的问题主要在于三星代工，而不是芯片本身。

总结来说，部分用户对骁龙888持有负面看法，主要是基于它在过去某些方面的不足；但也有用户认为其在特定条件下表现可以接受。<ans>
```

## Reference
https://www.elastic.co/guide/en/elasticsearch/reference/current/quickstart.html