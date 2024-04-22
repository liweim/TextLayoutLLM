import re
import numpy as np
from transformers import AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)
import argparse
from loguru import logger
import os
from os.path import join
import torch
import bitsandbytes as bnb
from collections import defaultdict
import glob

from component.collator import SFTDataCollator
from component.dataset import SFTDataset, ChatGLM2SFTDataset, Llama2SFTDataset
from component.argument import QLoRAArguments
from component.trainer import LoRATrainer
from component.loss import TargetLMLoss


def verify_model_dtype(model):
    dtype2param_num = defaultdict(int)
    dtype2param_name = defaultdict(list)
    dtype2trainable_param_num = defaultdict(int)
    dtype2trainable_param_name = defaultdict(list)
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # total = 0
    # print('verify all params of the model')
    # for k, v in dtype2param_num.items():
    #     total += v
    # for k, v in dtype2param_num.items():
    #     print(k, v, v / total)
    # for k, v in dtype2trainable_param_name.items():
    #     print(k, v)

    # print()
    # print('verify trainable params the model')
    # total_trainable = 0
    # for k, v in dtype2trainable_param_num.items():
    #     total_trainable += v
    # for k, v in dtype2trainable_param_num.items():
    #     print(k, v, v / total_trainable)
    # for k, v in dtype2trainable_param_num.items():
    #     print(k, v)


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/qlora/chatglm3-6b-sft-qlora.json', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    try:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
    except:
        pass
    # logger.add(join(training_args.output_dir, 'train.log'))
    # logger.info("train_args:{}".format(training_args))
    set_seed(training_args.seed)
    return args, training_args


def change_optimizer(training_args):
    # change optimizer from paged_adamw_32bit to adamw_torch
    optimizer_paths = glob.glob(training_args.output_dir + '/checkpoint*/optimizer.pt')
    if len(optimizer_paths) > 0:
        optimizer_path = optimizer_paths[0]
        state_dict = torch.load(optimizer_path)
        for i in state_dict["state"].keys():
            if "exp_avg" in state_dict["state"][i]:
                return
            exp_avg = state_dict["state"][i].pop("state1")
            exp_avg_sq = state_dict["state"][i].pop("state2")
            state_dict["state"][i]["exp_avg"] = exp_avg
            state_dict["state"][i]["exp_avg_sq"] = exp_avg_sq

        for i in range(len(state_dict["param_groups"])):
            state_dict["param_groups"][i]["amsgrad"] = False
            state_dict["param_groups"][i]["foreach"] = None
            state_dict["param_groups"][i]["maximize"] = False
            state_dict["param_groups"][i]["capturable"] = False
            state_dict["param_groups"][i]["differentiable"] = False
            state_dict["param_groups"][i]["fused"] = None

        torch.save(state_dict, optimizer_path)


def init_components(args, training_args):
    logger.info('Initializing components...')
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # device_map = "auto"
    # # if we are in a distributed setting, we need to set the device map and max memory per device
    # if os.environ.get('LOCAL_RANK') is not None:
    #     local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    #     device_map = {'': local_rank}

    change_optimizer(training_args)

    training_args.ddp_find_unused_parameters = False
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False if model.config.model_type == 'llama' else True
    )
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    elif tokenizer.__class__.__name__ != 'ChatGLMTokenizer':
        assert tokenizer.eos_token_id is not None
        assert tokenizer.bos_token_id is not None
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    # if tokenizer.pad_token_id == tokenizer.eos_token_id and tokenizer.unk_token_id is not None:
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    # if tokenizer.pad_token_id == tokenizer.eos_token_id:
    #     raise Exception('pad_token_id should not be equal to eos_token_id')

    # casts all the non int8 modules to full precision (fp32) for stability
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
    target_modules = find_all_linear_names(model)
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    verify_model_dtype(model)

    loss_func = TargetLMLoss(ignore_index=-100)

    if model.config.model_type == 'chatglm':
        train_dataset = ChatGLM2SFTDataset(args.train_file, tokenizer, args.max_seq_length)
        # eval_dataset = ChatGLM2SFTDataset(args.eval_file, tokenizer, args.max_seq_length)
    elif model.config.model_type == 'llama':
        train_dataset = Llama2SFTDataset(args.train_file, tokenizer, args.max_seq_length)
        # eval_dataset = Llama2SFTDataset(args.eval_file, tokenizer, args.max_seq_length)
    else:
        train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length)
        # eval_dataset = SFTDataset(args.eval_file, tokenizer, args.max_seq_length)
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        # tokenizer=tokenizer,
        data_collator=data_collator,
        compute_loss=loss_func,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=25)]
    )
    return trainer


def main():
    args, training_args = setup_everything()
    trainer = init_components(args, training_args)
    logger.info("*** starting training ***")
    # todo resume from checkpoint
    # https://github.com/huggingface/transformers/issues/24252
    checkpoints = glob.glob(training_args.output_dir + '/checkpoint*')
    if len(checkpoints) > 0:
        steps = [int(re.findall(r'checkpoint-(.*)', checkpoint)[0]) for checkpoint in checkpoints]
        max_step = np.argmax(steps)
        train_result = trainer.train(resume_from_checkpoint=checkpoints[max_step])
    else:
        train_result = trainer.train()
    final_save_path = join(training_args.output_dir, 'final')
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(training_args.output_dir)


if __name__ == "__main__":
    main()


