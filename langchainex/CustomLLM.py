from langchain.callbacks import CallbackManager
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel
from typing import Any, List, Mapping, Optional

class CustomLLM(LLM):
    
    tokenizer: int
    model: int
    history: list

    def setModel(self):
      tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
      self.tokenizer = tokenizer
      model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
      model = model.quantize(8)
      self.model = model
      self.history = []
        
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        if(len(prompt) >= 2048):
            prompt = prompt[len(prompt) - 2048:]
        response, __ = self.model.chat(self.tokenizer, prompt, top_p=0.3, history=self.history)
        print(response)
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.history}

