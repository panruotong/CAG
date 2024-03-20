import argparse
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import os
import openai
from typing import List, Dict, Optional, Sequence

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_chain,
    wait_fixed
)
openai.api_base = ""
openai.api_key = ""

class OpenAIInferencer:
    def __init__(
        self,
        model:str,
        api_key:str,
        api_base:str=None,
        temperature:float=0,
        max_tokens:int=1024,
        request_timeout=120,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        openai.api_key = api_key
        if api_base is not None:
            openai.api_base = api_base

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                        [wait_fixed(5) for i in range(2)] +
                        [wait_fixed(10)]))
    def inference(self, prompt, stop=None):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop
        )
        return response['choices'][0]['text'].strip()

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                        [wait_fixed(5) for i in range(2)] +
                        [wait_fixed(10)]))
    def chat_inference(self, messages: List[Dict]):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                request_timeout=self.request_timeout,
            )
        except Exception as e:
            if str(e).startswith("json"):
                print(messages)
            print(e)
        return response['choices'][0]['message']['content'].strip()
    
    def multiprocess_chat_inference(self, 
                                    batch_of_messages: List[List[Dict]], 
                                    num_workers:int=8):

        pool = ThreadPool(num_workers)
        all_completions = list(
            tqdm(pool.imap(self.chat_inference, batch_of_messages), 
                 total=len(batch_of_messages))
        )

        return all_completions

    def tokenize(self, prompt:str) -> List:
        enc = tiktoken.encoding_for_model(self.model)
        return enc.encode(prompt)

def evalChatgpt(eval_data, model_name, type_, temperature, f):
    if "cred" in type_:
        system = "You are an assistant who can answer questions based on the given passages. Each passage has a credibility score that indicates the relevance and accuracy of the passage to the question. Your answer will need to combine multiple passages and their credibility scores."
    else:
        system = "You are an accurate and reliable AI assistant that can answer questions with the help of external documents."
    #gpt-3.5-turbo-16k-0613
    openai_infer = OpenAIInferencer(model=model_name, temperature = temperature, api_key = openai.api_key, api_base = openai.api_base)
    batch_size = 20
    for i in range(0, len(eval_data), batch_size):
        message_batch = []
        answers = []
        for j in range(batch_size):
            try:
                example = eval_data[i+j]
                conversations = example["conversations"]
                for message in conversations:
                    if message["from"] == "human":
                        human_text = message["value"]
                    else:
                        answer = message["value"]
                messages = [{"role":"system","content":system}, {"role":"user","content": human_text}]
                message_batch.append(messages)
                answers.append(answer)
            except IndexError:
                break
        preds = openai_infer.multiprocess_chat_inference(message_batch, num_workers=4)
        for pred, answer in zip(preds, answers):
            f.write(json.dumps({"output": pred, "golden": answer}))
            f.write("\n")