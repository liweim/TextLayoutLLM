from transformers import AutoTokenizer
import torch

import sys
sys.path.append("../../")
from component.utils import ModelUtils


def main():
    model_name_or_path = 'YeungNLP/firefly-baichuan-13b'
    adapter_name_or_path = None

    load_in_4bit = False
    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0
    device = 'cuda'
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False if model.config.model_type == 'llama' else True
    )
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    text = input('User：')
    while True:
        text = text.strip()
        if model.config.model_type == 'chatglm':
            text = '[Round 1]\n\n问：{}\n\n答：'.format(text)
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        else:
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
            eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
            input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(tokenizer.eos_token, "").strip()
        print("Firefly：{}".format(response))
        text = input('User：')


if __name__ == '__main__':
    main()
