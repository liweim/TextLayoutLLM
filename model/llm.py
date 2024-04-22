# This Python file uses the following encoding: utf-8

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
import ssl
import tiktoken
from langchain.llms import OpenAI
import base64
from langchain.callbacks import get_openai_callback
import torch
import re
import os
import os.path as osp
import glob
import numpy as np

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk

nltk.download('punkt')
openai_api_key = ""
os.environ["OPENAI_API_KEY"] = openai_api_key

root = '/root/path/to/LLM/weights'
cur_dir = os.path.dirname(os.path.abspath(__file__))


class LlmBase():
    def __init__(self, model_path, temperature=0.1, device='auto'):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device,
                                                          trust_remote_code=True)
        self.temperature = max(0.1, temperature)
        self.device = device

    def __call__(self, question):
        inputs = self.tokenizer(question, return_tensors='pt')
        inputs = inputs.to(self.device)
        pred = self.model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.1, temperature=self.temperature)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response[len(question):-3]
        return response


class ModelUtils(object):
    @classmethod
    def load_model(cls, model_name_or_path, load_in_4bit=False, adapter_name_or_path=None, device='auto'):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map=device,
            quantization_config=quantization_config
        )

        if adapter_name_or_path is not None:
            model = PeftModel.from_pretrained(model, adapter_name_or_path)

        return model


class LlmChat():
    def __init__(self, model_name_or_path, adapter_name_or_path, temperature=0.1, device='auto'):
        if adapter_name_or_path is not None:
            if 'checkpoint' not in adapter_name_or_path:
                checkpoints = glob.glob(adapter_name_or_path + '*/checkpoint*')
                steps = [int(re.findall(r'checkpoint-(.*)', checkpoint)[0]) for checkpoint in checkpoints]
                min_step = np.argmin(steps)
                adapter_name_or_path = checkpoints[min_step]
            print(f'loaded model from {adapter_name_or_path}')
        else:
            print(f'loaded model from {model_name_or_path}')
        self.model_name_or_path = model_name_or_path
        self.adapter_name_or_path = adapter_name_or_path
        self.temperature = max(0.1, temperature)
        self.device = device

        if 'baichuan2-7b-chat' in model_name_or_path or 'baichuan2-13b-chat' in model_name_or_path:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.generation.utils import GenerationConfig
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False,
                                                           trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device,
                                                              torch_dtype=torch.bfloat16, trust_remote_code=True)
            self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
            self.model.generation_config.temperature = self.temperature
            self.model.generation_config.max_new_tokens = 512
        else:
            from transformers import AutoTokenizer
            load_in_4bit = False

            self.model = ModelUtils.load_model(
                model_name_or_path,
                load_in_4bit=load_in_4bit,
                adapter_name_or_path=adapter_name_or_path,
                device=device
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_fast=False if self.model.config.model_type == 'llama' else True
            )
            if tokenizer.__class__.__name__ == 'QWenTokenizer':
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token_id = tokenizer.eod_id
                tokenizer.eos_token_id = tokenizer.eod_id
            self.tokenizer = tokenizer
            self.temperature = self.temperature

    def __call__(self, question, max_new_tokens=512, top_p=0.85, repetition_penalty=1.05):
        if 'baichuan2-7b-chat' in self.model_name_or_path or 'baichuan2-13b-chat' in self.model_name_or_path:
            messages = []
            messages.append({"role": "user", "content": question})
            response = self.model.chat(self.tokenizer, messages)
            return response
        else:
            text = question.strip()
            if self.model.config.model_type == 'chatglm':
                text = '[Round 1]\n\nquestion：{}\n\nanswer: '.format(text)
                input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
                    'cuda')
            else:
                input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
                    'cuda')
                bos_token_id = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long).to('cuda')
                eos_token_id = torch.tensor([[self.tokenizer.eos_token_id]], dtype=torch.long).to('cuda')
                input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                    top_p=top_p, temperature=self.temperature, repetition_penalty=repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            outputs = outputs.tolist()[0][len(input_ids[0]):]
            response = self.tokenizer.decode(outputs)
            response = response.strip().replace(self.tokenizer.eos_token, "").strip()
            if self.model.config.model_type == 'llama':
                response = re.sub(r'^(s |<s> #|<s>)', '', response).strip()
            return response


class LocalLLM(LLM):
    url: str = 'http://0.0.0.0:8088'
    model_name: str = 'chatGLM'
    temperature: float = 0

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if isinstance(stop, list):
            stop = stop + ["Observation:"]

        if self.model_name == 'chatGLM-6b':
            response = requests.post(
                self.url,
                json={
                    "prompt": prompt,
                    "histroy": [],
                    "temperature": self.temperature,
                    "stop": stop,
                },
            )
            response.raise_for_status()
            return response.json()["response"]
        elif self.model_name == 'baichuan2-13b':
            response = requests.post(
                self.url,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": 512,
                    "use_cache": False
                },
            )
            response.raise_for_status()
            return response.json()["response"]
        elif self.model_name == 'vicuna-33b' or self.model_name == 'llama-13b':
            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "stop": stop
                }
            )
            response.raise_for_status()
            response = response.json()['choices'][0]['message']['content']
            return response
        else:
            print(f'not supported model: {self.model_name}')

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}


def load_llm(model_name, model_type='encoder-decoder', temperature=0,
             use_cache=False, device='auto'):
    if use_cache:
        import langchain
        from langchain.cache import SQLiteCache
        langchain.llm_cache = SQLiteCache(database_path="tmp/.langchain.db")

    tokenizer = None
    max_length = 512
    if model_name == 'gpt-3.5':
        llm = OpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=512)
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        max_length = 4000
    elif model_name == 'gpt-4':
        llm = OpenAI(model_name="gpt-4", temperature=temperature)
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        max_length = 8000
    elif model_name.startswith('baichuan2'):
        from transformers import AutoTokenizer
        max_length = 4000

        if model_name == 'baichuan2-13b':
            llm = LlmChat(root + 'baichuan2-13b-chat', None, temperature=temperature, device=device)
        elif model_name == 'baichuan2-13b-base':
            llm = LlmBase(root + 'baichuan2-13b-base', temperature=temperature, device=device)
        elif model_name.startswith('baichuan2-13b-base-'):
            llm = LlmChat(root + 'baichuan2-13b-base', root + model_name, temperature=temperature, device=device)
        elif model_name.startswith('baichuan2-13b-'):
            llm = LlmChat(root + 'baichuan2-13b-chat', root + model_name, temperature=temperature, device=device)
        elif model_name == 'baichuan2-7b':
            llm = LlmChat(root + 'baichuan2-7b-chat', None, temperature=temperature, device=device)
        elif model_name == 'baichuan2-7b-base':
            llm = LlmBase(root + 'baichuan2-7b-base', temperature=temperature, device=device)
        elif model_name.startswith('baichuan2-7b-'):
            llm = LlmChat(root + 'baichuan2-7b-chat', root + model_name, temperature=temperature, device=device)
        else:
            raise Exception('not supported model')
        tokenizer = llm.tokenizer
    elif model_name.startswith('chatglm3'):
        max_length = 7500
        if model_name == 'chatglm3-6b':
            llm = LlmChat(root + 'chatglm3-6b-chat', None, temperature=temperature, device=device)
        elif model_name == 'chatglm3-6b-base':
            llm = LlmBase(root + 'chatglm3-6b-base', temperature=temperature, device=device)
        elif model_name.startswith('chatglm3-6b-'):
            llm = LlmChat(root + 'chatglm3-6b-chat', root + model_name, temperature=temperature, device=device)
        else:
            raise Exception('not supported model')
        tokenizer = llm.tokenizer
    elif model_name.startswith('llama2'):
        max_length = 4000
        if model_name == 'llama2-7b':
            llm = LlmChat(root + 'llama2-7b-chat', None, temperature=temperature, device=device)
        elif model_name == 'llama2-7b-base':
            llm = LlmBase(root + 'llama2-7b-base', temperature=temperature, device=device)
        elif model_name.startswith('llama2-7b-'):
            llm = LlmChat(root + 'llama2-7b-chat', root + model_name, temperature=temperature, device=device)
        elif model_name == 'llama2-13b':
            llm = LlmChat(root + 'llama2-13b-chat', None, temperature=temperature, device=device)
        elif model_name == 'llama2-13b-base':
            llm = LlmBase(root + 'llama2-13b-base', temperature=temperature, device=device)
        elif model_name.startswith('llama2-13b-'):
            llm = LlmChat(root + 'llama2-13b-chat', root + model_name, temperature=temperature, device=device)
        else:
            raise Exception('not supported model')
        tokenizer = llm.tokenizer
    else:
        raise Exception('Not supported LLM')

    return llm, tokenizer, max_length


def limit_token_length(tokenizer, max_length, prompt_template, ori_context, question=''):
    context = ori_context
    exceed_limit = False
    if tokenizer is not None:
        tokens = tokenizer.encode(context)
        num_tokens = len(tokens)
        ratio = num_tokens / len(context)
        num_prompt_tokens = len(tokenizer.encode(prompt_template))
        limit = max_length - num_prompt_tokens
        if num_tokens > limit:
            if question != '' and question in context:
                half_limit = int(limit / 2)
                idx = int(context.index(question) * ratio)
                tokens = tokens[max(0, idx - half_limit):min(len(tokens), idx + half_limit)]
            else:
                tokens = tokens[:limit]
            context = tokenizer.decode(tokens)
            print(
                f'context length: {len(ori_context)}, limit to {len(context)} ({len(context) / len(ori_context) * 100}%)')
            # print(f'input after limiting token length:\n{context}')
            exceed_limit = True
    return context, exceed_limit


if __name__ == '__main__':
    llm = load_llm('chatglm3-6b-base')[0]
    print(llm('how are you'))
    # llm = OpenAI(model_name="gpt-3.5")
    # with get_openai_callback() as cb:
    #     print(llm("Tell me a joke"))
    #     print(cb)
