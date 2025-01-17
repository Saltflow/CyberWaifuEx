from waifu.llm.Brain import Brain
from waifu.llm.VectorDB import VectorDB
from langchain.chat_models import ChatOpenAI
from typing import Any, List, Mapping, Optional
from langchain.schema import BaseMessage
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


class GLM(Brain):
    def __init__(self, api_key: str,
                 name: str,
                 stream: bool=False,
                 callback=None,
                 model: str="THUDM/chatglm2-6b-int4",
                 is_cuda = False,
                 proxy: str=''):
        self.llm = CustomLLM(tokenizer=1, model=1, history=[])
        # self.llm_nonstream = ChatOpenAI(openai_api_key=api_key, model_name=model)
        self.llm.setModel(model, is_cuda)
        # self.embedding = OpenAIEmbeddings(openai_api_key=api_key)

        self.embedding = HuggingFaceEmbeddings()
        # self.embedding = STEmbedding()
        self.vectordb = VectorDB(self.embedding, f'./memory/{name}.csv')



    def think(self, messages: List[BaseMessage]):
        return self.llm(messages)


    def think_nonstream(self, messages: List[BaseMessage]):
        return self.llm(messages)


    def store_memory(self, text: str | list):
        '''保存记忆 embedding'''
        self.vectordb.store(text)


    def extract_memory(self, text: str, top_n: int = 10):
        '''提取 top_n 条相关记忆'''
        return self.vectordb.query(text, top_n)
    
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel
from typing import Any, List, Mapping, Optional

class CustomLLM(LLM):
    
    tokenizer: int
    model: int
    history: list

    def setModel(self, model, is_cuda):
      tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    #   tokenizer = AutoTokenizer.from_pretrained("GLMM", trust_remote_code=True)
      self.tokenizer = tokenizer
      # model = AutoModel.from_pretrained(model, trust_remote_code=True).float()
      if(is_cuda):
          model = AutoModel.from_pretrained(model, trust_remote_code=True).half().cuda()
          model = model.quantize(8)
      else:
          model = AutoModel.from_pretrained(model, trust_remote_code=True).float()
    #   model = AutoModel.from_pretrained("GLMM", trust_remote_code=True).half()
    #   
      self.model = model
      self.history = []
        
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        if(len(self.history) == 0):
          content = ''.join(list(map(lambda x: x.content, prompt)))

          response, history = self.model.chat(self.tokenizer, content, top_p=0.3)
        else:
          content = prompt[-1].content
          response, history = self.model.chat(self.tokenizer, content, top_p=0.3, history=self.history)
        print(response)
        self.history = history
        return response
    
    def _flow_call(
            self, prompt
    ):
        response, _ = self.model.flow_chat(self.tokenizer, prompt, top_p=0.3, history=self.history)
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.history}

