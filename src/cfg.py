class Config:

    def __init__(self,
                 llm_model_name="Qwen-V1-8K-Chat",
                 llm_prompt="You are a helpful assistant.",
                 llm_temperature=0.9,
                 llm_top_p=0.7,
                 llm_max_tokens=2048,
                 sr_model_name="base",
                 sr_language="en",
                sr_verbose=False):
        self.llm_model_name = llm_model_name