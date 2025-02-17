# 运行 RAG LLM
from ragllm import RAGLLM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama_url", type=str)
    parser.add_argument("--ollama_model", type=str)
    parser.add_argument("--chat_id", type=int, default=0)
    args = parser.parse_args()

    llm = RAGLLM(args.ollama_url, args.ollama_model)
    while True:
        q = input("Query (input 'exit' to exit): ")
        try:
            if q.lower() == "exit":
                break
            print(llm.run(q, args.chat_id))

        except KeyboardInterrupt as e:
            break
        except Exception as e:
            print(e);break
        
    llm.search_api.close()