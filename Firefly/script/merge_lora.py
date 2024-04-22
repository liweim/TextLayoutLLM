from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
"""
使用该脚本，将lora的权重合并到base model中
"""


def merge_lora_to_base_model():
    model_name_or_path = '/weiming_new/llm/llama2-13b-chat'
    adapter_name_or_path = '/weiming_new/llm/llama2-13b-code_200k_3/checkpoint-9000'
    save_path = '/weiming_new/llm/llama2-13b-code_200k_3-merge'

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'}, trust_remote_code=True)
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
