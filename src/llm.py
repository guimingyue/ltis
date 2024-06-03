import random
import dashscope
from http import HTTPStatus

class Llm:
    def __init__(self, initial_prompt) -> None:
        pass

    def call_with_messages(self, text: str):
        return text, True

class Qwen(Llm):

    def __init__(self, model= "qwen-turbo", prompt = "You are a helpful assistant." ,
                 temperature=0.9, top_p=0.7, max_tokens=2048) -> None:
        print("init prompt: " + prompt)
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = False
        self.messages=[{'role': 'system', 'content': prompt}]
        #self.stop = ["\nHuman:", "\n\nHuman:"]

    '''
    you need to save api key
    '''
    def call_with_messages(self, text):
        self.messages.append({'role': 'user', 'content': text})
        response = dashscope.Generation.call(
            model=self.model_name,
            messages=self.messages,
            seed=random.randint(1, 10000),
            result_format='message',  # set the result to be "message" format.
        )
        if response.status_code == HTTPStatus.OK:
            result = response.output.choices[0]
            content = result['message']['content']
            self.messages.append({'role': result['message']['role'], 'content': content})
            return content, True
        else:
            self.messages[:-1]
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            return 'EOF ERROR', False