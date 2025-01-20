# 处理爬取的 Bilibili 评论数据

import json
import os
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from embedding import EmbeddingModel
from pymilvus import MilvusClient

def convert_timestamp(timestamp):
    """
    UTC+8时间戳转换
    """
    utc_time = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(timezone(timedelta(hours=8)))
    formatted_time = utc_time.strftime("%Y年%m月%d日")
    return formatted_time

def get_comment_detail():
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
            comment_detail[item["comment_id"]] = {
                "comment_id": item["comment_id"],                 # 评论 id
                "viedo_info": video_detail[item["video_id"]],   # 视频信息
                "create_time": item["create_time"],             # 评论时间
                "nickname": item["nickname"],                       # 评论者昵称
                "parent_comment_id": item["parent_comment_id"],                 # 父级评论 id
                "content": item["content"],                     # 评论内容
            }

    # 写入父级评论内容
    print("Processing parent comment details...")
    for k,v in tqdm(comment_detail.items()):
        # 父级评论不存在时跳过
        if v["parent_comment_id"] in comment_detail.keys():
            comment_detail[k]["parent_content"] = comment_detail[v["parent_comment_id"]]["content"]
        else:
            comment_detail[k]["parent_content"] = "None"

    # 构造结构化文档
    # 文档结构为： "[ID: xxx][视频发布时间：xxxx年xx月xx日][视频标题：xxx][视频简介：xxx][评论时间：xxxx年xx月xx日][评论者昵称：xxx][父级评论：xxx][评论内容：xxx]"
    comments = []
    for k,v in tqdm(comment_detail.items()):
        v_create_time = convert_timestamp(v["viedo_info"]["create_time"])
        v_title = v["viedo_info"]["title"]
        v_desc = v["viedo_info"]["desc"]
        c_create_time = convert_timestamp(v["create_time"])
        c_nickname = v["nickname"]
        c_parent_content = v["parent_content"]
        c_content = v["content"]
        c_id = v["comment_id"]

        mesg = f"[ID: {c_id}][视频发布时间: {v_create_time}][视频标题: {v_title}][视频简介: {v_desc}][评论时间: {c_create_time}][评论者昵称: {c_nickname}][父级评论: {c_parent_content}][评论内容: {c_content}]"

        comments.append(mesg)

    return comments

def init_milvus(c_detail: list, chunk_size=50):
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

    tasks = [c_detail[i:i + chunk_size] for i in range(0, len(c_detail), chunk_size)]
    for i, t in enumerate(tqdm(tasks)):
        # 每处理完一个批次就异步插入数据
        result = embedding_model.get_embedding(t)
        insert_data = []
        for j, x in enumerate(result["text_embedding"]):
            insert_data.append({"id": i*chunk_size+j, "vector": x.tolist(), "text": t[j]})
        milvus_client.insert(collection_name=collection_name, data=insert_data)
    
    milvus_client.close()

if __name__ == '__main__':
    c_detail = get_comment_detail()
    init_milvus(c_detail)