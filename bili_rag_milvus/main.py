# RAG-Milvus
# 本地 Embedding Model + Qwen-plus 在线API模型 + 本地 Milvus 数据库
from glob import glob
from openai import OpenAI
from embedding import EmbeddingModel
import yaml
import os
from pymilvus import MilvusClient
import re

class BiliMilvusRAG:
    def __init__(self):
        """
        初始化参数
        """
        with open('config.yaml', 'r', encoding='utf-8') as file:
            self.cfg = yaml.safe_load(file)
        # 使用的平台名字
        self.p_name = self.cfg["used"]

        # 初始化向量数据库milvus
        db_file = "./data/milvus.db"
        if not os.path.exists(db_file):
            raise Exception(f"Please run prepare_data.py first.")
        
        self.milvus_client = MilvusClient(uri=db_file)
        self.collection_name = "bili_comments"  

        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=768,
            metric_type="IP",
            consistency_level="Strong",
        ) 
        # 初始化 Embedding 模型
        self.embedding = EmbeddingModel()

        # 初始化 LLM 模型
        self.llm = OpenAI(
            api_key=self.cfg["platform"][self.p_name]["api_key"], 
            base_url=self.cfg["platform"][self.p_name]["base_url"],
            )
        
    def run(self, question):
        # 采用类 Agent 的方法
        # 1 先询问 llm 问题，让 llm 检索自己的参数空间。如无法确定问题答案，需要返回某些特殊符号以提示本地去 milvus 数据库查询内容
        # 2 如果 llm 已经确定答案，则返回另一些特殊符号，结束问答

        unknow_symbol = "<unk>"
        answer_symbol = "<ans>"
        tips_symbol_s = "<tip>"
        tips_symbol_e = "</tip>"
        max_tries = 5

        system_prompt = f"""
        人类：你是一个人工智能助手，我会问你一些问题。首先你应该不依赖外部检索尝试给出问题的答案，如果不确定问题的答案，请返回 {unknow_symbol} 以提示本地去检索。如果答案确定，请返回 {answer_symbol} 以结束问答。
        你只有 {max_tries} 次机会尝试回答这个问题。每次结束回答且无法给出问题的答案时，你可以用 {tips_symbol_s} 来告诉我你不知道的部分，以提示我继续尝试本地检索。例如，你可以用 {tips_symbol_s} XXX {tips_symbol_e} 来包裹问题，其中XXX是你想进一步询问我的问题。记住，无论在什么情况下，你都需要返回之前所定义的三种特殊符号以结束问答。
        """

        # 结束符、尝试次数、llm提示
        fin, tries, tips, ans = False, 1, "", "LLM 无法回答此问题。"
        while not fin and tries <= max_tries:
            # 第一次问答
            if tries == 1:
                user_prompt = question
                context = ""
                
            llm_ans = self.chat(user_prompt, context, system_prompt)

            print(f"# 尝试次数: {tries}")
            print(f"# user_prompt: {user_prompt}")
            print(f"# tips: {tips}")
            print(f"# context: {context}")
            print(f"# answer: {llm_ans}")

            # 1 确定答案的情况
            if answer_symbol in llm_ans:
                ans = llm_ans
                fin = True
            # 2 不知道问题且有提示的情况
            elif tips_symbol_s in llm_ans:
                pattern = rf"{tips_symbol_s}(.*?){tips_symbol_e}"
                matches = re.findall(pattern, llm_ans)
                if len(matches) > 0:
                    tips = matches[0]
                    context = "".join(self.search_milvus(tips))
                else:
                    user_prompt = f"{tips_symbol_s}不合规范，无法检索问题，请重新回答以符合规范。"
                    context = ""
            # 3 不知道问题且没有提示的情况
            elif unknow_symbol in llm_ans:
                context = "".join(self.search_milvus(question))
            else:
                user_prompt = f"不存在{unknow_symbol}与{answer_symbol}标识，不合规范，无法检索问题，请重新回答以符合规范。"
                context = ""
            tries += 1

        return ans

    def search_milvus(self, question, limit=10):
        """
        到 milvus 数据库中检索内容
        """
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=self.embedding.get_embedding([question])["text_embedding"].tolist(),
            limit=limit,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],  # Return the text field
        )
  
        context = [x["entity"]["text"] for x in search_res[0]]
        return context

    def chat(self, question, context, system_prompt):
        """
        调用 OpenAI 的 API 进行聊天
        """
        if context != "":
            user_prompt = f"""
            使用以下包含在 <context> 标记中的信息片段来回答 <question> 标记中包含的问题。
            <context>
            {context}
            </context>
            <question>
            {question}
            </question>
            """
        else:
            user_prompt = question

        # 调用 API
        completion = self.llm.chat.completions.create(
            model=self.cfg["platform"][self.p_name]["model"],
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}],
            )
        
        ans = completion.choices[0].message.content
        return ans

if __name__ == "__main__":
    rag = BiliMilvusRAG()

    # 预设问题
    questions = [
        "B站用户对骁龙888的评价如何？",
        "B站用户相较于AMD和英伟达更喜欢哪个？",
        "B站用户在数码产品上有哪些爱好？",
        "我想配一台5000块的主机，B站用户会给我什么建议？"
    ]
    for q in questions:
        ans = rag.run(q)

    # while True:
    #     q = input("# 请输入问题：")
    #     if not q:
    #         break
    #     rag.run(q)




