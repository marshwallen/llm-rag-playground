# 处理爬取的 Bilibili 评论数据

import json
import os
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from es_client import ESClient
from embedding import EmbeddingModel

def convert_timestamp(timestamp):
    """
    UTC+8时间戳转换
    """
    utc_time = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(timezone(timedelta(hours=8)))
    formatted_time = utc_time.strftime("%Y-%m-%d")
    return formatted_time

def get_comment_detail():
    """
    处理获取的 Bilibili 评论数据
    """
    # 读取json文件内容
    contents_dir = [x for x in os.listdir("./data/") if "contents" in x and "json" in x]
    comments_dir = [x for x in os.listdir("./data/") if "comments" in x and "json" in x]

    # 视频信息
    print("Processing video details...")
    video_detail = {}
    for d in contents_dir:
        with open(f'./data/{d}', 'r', encoding='utf-8') as f:
            contents_json = json.load(f)
        for item in tqdm(contents_json):
            video_detail[item["video_id"]] = {
                "create_time": item["create_time"], # 视频发布时间
                "title": item["title"],             # 视频标题
                "desc": item["desc"],               # 视频简介
            }

    # 评论信息
    print("Processing comment details...")
    comment_detail = {}
    for d in comments_dir:
        with open(f'./data/{d}', 'r', encoding='utf-8') as f:
            comments_json = json.load(f)
        for item in tqdm(comments_json):
            video_create_time = convert_timestamp(video_detail[item["video_id"]]["create_time"])
            comment_create_time = convert_timestamp(item["create_time"])

            comment_detail[item["comment_id"]] = {
                "video_create_time": video_create_time,                 # 视频发布时间
                "video_title": video_detail[item["video_id"]]["title"], # 视频标题
                "video_desc": video_detail[item["video_id"]]["desc"],   # 视频简介
                "comment_create_time": comment_create_time,             # 评论时间
                "nickname": item["nickname"],                           # 评论者昵称
                "parent_comment_id": item["parent_comment_id"],         # 父级评论 id
                "content": item["content"],                             # 评论内容
            }
    
    # 写入父级评论内容
    print("Processing parent comment details...")
    for k,v in tqdm(comment_detail.items()):
        # 父级评论不存在时跳过
        if v["parent_comment_id"] in comment_detail.keys():
            comment_detail[k]["parent_content"] = comment_detail[v["parent_comment_id"]]["content"]
        else:
            comment_detail[k]["parent_content"] = "None"

    comments = []
    exclude = ["parent_comment_id"]
    for k,v in tqdm(comment_detail.items()):
        comments.append("".join([f"{k0}:{v0} " for k0, v0 in v.items() if k0 not in exclude]))

    return comments

def init_es(c_detail: list, chunk_size=50):
    """
    将评论数据 Embedding 后写入 ES 数据库
    """
    # 构造 Embedding model
    embedding_model = EmbeddingModel()

    # 初始化 ES 数据库
    es = ESClient()
    index = "bili_comments" # 索引名

    # 创建索引
    mapping = {
        "mappings": {
            "properties": {
                "text": {
                    "type": "text"
                },
                "vector": {
                    "type": "dense_vector",
                    "dims": 768
                },
            }
        }
    }
    if not es._client.indices.exists(index=index):
        es._client.indices.create(index=index, body=mapping)

    tasks = [c_detail[i:i + chunk_size] for i in range(0, len(c_detail), chunk_size)]
    for i, t in enumerate(tqdm(tasks)):
        # 每处理完一个批次就异步插入数据
        result = embedding_model.get_embedding(t)
        insert_data = {}
        for j, x in enumerate(result["text_embedding"]):
            insert_data[i*chunk_size+j] = {"text": t[j], "vector": x.tolist()}
        es.insert(index, insert_data)

    es.close()
    print("Done. Comment counts: ", len(c_detail))

if __name__ == '__main__':
    c_detail = get_comment_detail()
    init_es(c_detail)