# 1 使用 nlp_gte_sentence-embedding_chinese-base 模型构造 embedding 后的向量
# 2 得到向量表示后，写入 milvus 数据库

# reference: https://www.modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-base

from modelscope import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import shutil
import os

class EmbeddingModel:
    def __init__(self, sequence_length=512, model_id="iic/nlp_gte_sentence-embedding_chinese-base"):
        """
        embedding 模型
        获得 txt 文本中每一行语料的向量表示
        """
        self.model_id = model_id

        if not os.path.exists(f"./model/{model_id}"):
            snapshot_download(model_id, cache_dir="./model", revision='master')
            shutil.rmtree("./model/._____temp")

        self.pipeline_se = pipeline(Tasks.sentence_embedding,
                            model=f"./model/{model_id}",
                            sequence_length=sequence_length
                            ) # sequence_length 代表最大文本长度，默认值为128

    def get_embedding(self, data: list):
        """
        推理函数
        当输入仅含有soure_sentence时,会输出source_sentence中每个句子的向量表示
        """
        if self.model_id == "iic/nlp_gte_sentence-embedding_chinese-base":
            return self.pipeline_se(input={"source_sentence": data})
        
        raise Exception("Not supported model.")

    def get_embedding_gte(self, source_sentence: list, sentences_to_compare: list):
        """
        gte 推理函数
        当输入包含“soure_sentence”与“sentences_to_compare”时,会输出source_sentence中首个句子与sentences_to_compare中每个句子的向量表示,
        以及source_sentence中首个句子与sentences_to_compare中每个句子的相似度
        """
        if self.model_id != "iic/nlp_gte_sentence-embedding_chinese-base":
            raise Exception("Model must be iic/nlp_gte_sentence-embedding_chinese-base")
        
        return self.pipeline_se(input={"source_sentence": source_sentence, "sentences_to_compare": sentences_to_compare})