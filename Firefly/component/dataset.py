import json
from loguru import logger
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']

        utterances = []
        for x in conversation:
            utterances.append(x['human'])
            utterances.append(x['assistant'])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        input_ids = [self.bos_token_id]
        target_mask = [0]
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + [self.eos_token_id])
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id) + 1)
            else:
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class Llama2SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']
        prompt_format = '''<s>[INST] <<SYS>>
{instruction}<</SYS>>

{input} [/INST]'''

        utterances = []
        for x in conversation:
            human = prompt_format.format(instruction=x['human'], input='')
            utterances.append(human)
            utterances.append(x['assistant'])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        input_ids = [self.bos_token_id]
        target_mask = [0]
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + [self.eos_token_id])
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id) + 1)
            else:
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class ChatGLM2SFTDataset(SFTDataset):

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']
        input_format = '[Round {}]\n\n问：{}\n\n答：'
        target_format = '{}'

        utterances = []
        for i, x in enumerate(conversation):
            human = input_format.format(i+1, x['human'])
            assistant = target_format.format(x['assistant'])
            utterances += ([human, assistant])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        input_ids = []
        target_mask = []
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += utterances_id
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id))
            else:
                input_ids += [self.eos_token_id]
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs

