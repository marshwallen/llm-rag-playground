from es_client import ESClient
from openai import OpenAI
from embedding import EmbeddingModel
import yaml
import re

class BiliESRAG:
    def __init__(self):
        with open('config.yaml', 'r', encoding='utf-8') as file:
            self.cfg = yaml.safe_load(file)
        # 使用的平台名字
        self.p_name = self.cfg["used"]

        # 启动 ES 客户端
        self.es = ESClient()
        # print("ES Data sample: ", self.es.search(index="bili_comments", body={"size":5}))
        self.index = "bili_comments"

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
                    t_vector = self.embedding.get_embedding([tips])["text_embedding"].tolist()[0]
                    context = self.es_resp_postprocess(
                                self.es.search(
                                    self.index,
                                    self.build_es_query(t_vector)
                                    )
                                )
                else:
                    user_prompt = f"{tips_symbol_s}不合规范，无法检索问题，请重新回答以符合规范。"
                    context = ""
            # 3 不知道问题且没有提示的情况
            elif unknow_symbol in llm_ans:
                q_vector = self.embedding.get_embedding([question])["text_embedding"].tolist()[0]
                context = self.es_resp_postprocess(
                                self.es.search(
                                    self.index,
                                    self.build_es_query(q_vector)
                                    )
                                )
            else:
                user_prompt = f"不存在{unknow_symbol}与{answer_symbol}标识，不合规范，无法检索问题，请重新回答以符合规范。"
                context = ""
            tries += 1

        return ans
    
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
    
    def build_es_query(self, question: str, size=20):
        """
        使用 match 或 more_like_this 查询，从ES中检索与问题相似的文档
        """
        query_body = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {
                            "query_vector": question
                        }
                    }
                }
            },
            "size": size
        }

        return query_body

    def es_resp_postprocess(self, response):
        """
        对ES的查询结果进行后处理，返回结果
        """
        hits = response["hits"]["hits"]
        if len(hits) == 0:
            return ""
        
        result = "\n".join([x["_source"]["text"] for x in hits])
        return result

if __name__ == '__main__':
    rag = BiliESRAG()

    # 预设问题
    questions = [
        "B站用户对骁龙888的评价如何？",
        "B站用户相较于AMD和英伟达更喜欢哪个？",
        "B站用户在数码产品上有哪些爱好？",
        "我想配一台5000块的主机，B站用户会给我什么建议？"
    ]
    for q in questions:
        ans = rag.run(q)