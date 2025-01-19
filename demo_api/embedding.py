import yaml
import dashscope
from http import HTTPStatus
from openai import OpenAI
from utils import config

def emb_text(p_name, text):
    # 文本到嵌入向量(embedding)
    cfg = config()

    api_key = cfg["platform"][p_name]["api_key"]

    # 调用阿里云百炼平台通用多模态向量模型
    if p_name == "Qwen":
        dashscope.api_key = api_key
        input = [{'text': text}]
        resp = dashscope.MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=input
        )

        if resp.status_code == HTTPStatus.OK:
            embedding = resp.output["embeddings"][0]["embedding"]
            return embedding
        else:
            raise Exception("Error in embedding")

    if p_name == "OpenAI":
        client = OpenAI(api_key=api_key)
        return (
            client.embeddings.create(input=text, model="text-embedding-3-small")
            .data[0]
            .embedding
        )