from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from searchapi import BochaaiSearchAPI
from sentence_transformers import CrossEncoder
import re

# 自定义搜索工具实现
class RAGLLM:
    def __init__(self, ollama_url, ollama_model):
        """
        智能 RAG: 分层检索架构设计
        graph TD
            A[用户提问] --> B{本地检索}
            B -->|结果存在| C[相关性评估]
            B -->|无结果| D[网络检索]
            C -->|相关| E[生成回答]
            C -->|不相关| D
            D --> F[合并结果]
            F --> E
        """
        # 初始化 Web 搜索工具
        self.search_api = BochaaiSearchAPI()

        # 初始化重排序模型
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 初始化 Ollama 模型
        model = ChatOllama(
            model=ollama_model,
            temperature=0.7,
            base_url=ollama_url
        )

        # Initialize memory to persist state between graph runs
        checkpointer = MemorySaver()
        self.app = create_react_agent(model, tools=[], checkpointer=checkpointer)
    
    def _retrieve(self, query:str, top_k=5) -> str:
        """
        本地/网络检索函数的具体实现
        """
        # 本地检索
        local_docs = self.search_api.search(query, local=True, limit=top_k*2)[:top_k*2]
        # 相关性过滤
        relevant_docs = self._relevance_check(query, local_docs)
        
        # 判断是否需要网络检索
        if len(relevant_docs) < top_k:
            web_docs = self.search_api.search(query, local=False, limit=top_k)
            combined = self._merge_results(relevant_docs, web_docs, query)
            return "".join(combined[:top_k])
        
        return "".join(relevant_docs[:top_k])
    
    def _relevance_check(self, query:str, docs:list[str], threshold=0.95):
        """使用交叉编码器进行精细相关性评估，并确保查询的关键信息在文档中存在"""
        if len(docs) == 0:
            return []
        
        # 提取查询中的关键数字
        match = re.search(r'\d+', query)
        if match:
            required_number = match.group()
        else:
            required_number = None

        pairs = [[query, doc] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        relevant_docs = []
        for doc, score in zip(docs, scores):
            if score > threshold:
                # 检查文档中是否包含所需的数字
                if required_number and required_number not in doc:
                    continue  # 如果文档不包含所需数字，则跳过
                relevant_docs.append(doc)
        
        return relevant_docs

    def _merge_results(self, local:list[str], web:list[str], query:str):
        """混合排序策略
        本地结果优先，但高质量网络结果可以提升排名
        """
        scored_docs = []
        for doc in local:
            scored_docs.append((doc, 1.0))  # 本地结果基础分
            
        for doc in web:
            score = self.reranker.predict([[query, doc]])[0]
            scored_docs.append((doc, score))
            
        # 按综合分排序
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]
    
    def run(self, query:str, thread_id:int, online=True) -> str:
        """断网或联网+本地数据库检索 RAG Chat
        """
        if online:
            search = self._retrieve(query)
        else:
            search = ""

        final_state = self.app.invoke(
            {"messages": [
                {"role": "user", "content": f"请回答主要问题：{query}。此外，你有以下补充信息：{search}。"}
                ]},
            config={"configurable": {"thread_id": thread_id}}
        )
        return final_state["messages"][-1].content
