import sys

project_root = '..'
sys.path.insert(0, project_root)

import os
from model.llm import load_llm, limit_token_length
from utils.metric import *


def compute_metrics(decoded_preds, decoded_labels):
    score_dict = {"accuracy": [], "rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": [], 'recall': []}

    no_std_ans_cnt, pred_lens, label_lens = 0, 0.0, 0.0
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_lens += len(pred)
        label_lens += len(label)
        if label == '':
            no_std_ans_cnt += 1
            continue

        bleu_score = compute_bleu([label], pred)
        result = compute_rouge(label, pred)
        for k, v in result.items():
            score_dict[k].append(v["f"])
        score_dict["bleu-4"].append(bleu_score)
        score_dict["accuracy"].append(float(len(label) != 0 and pred[:len(label)] == label))
        score_dict['recall'].append(compute_recall(label, pred))

    res_dict = {k: round(float(np.mean(v)), 4) for k, v in score_dict.items()}
    res_dict['samples'] = len(decoded_labels)
    res_dict['no_std_ans_cnt'] = no_std_ans_cnt
    return res_dict


def evaluate(path, save_path):
    with open(path, 'r', encoding='utf-8') as f:
        eval_set_result = json.load(f)

    cmrc_data = {}
    cmrc_data['all'] = {}
    cmrc_data['all']['decoded_preds'] = []
    cmrc_data['all']['decoded_labels'] = []
    for dic in eval_set_result.values():
        class_name = dic['source']
        decoded_pred = dic['answer']
        decoded_label = dic['output']
        if class_name == 'game':
            decoded_label = eval(re.findall(r'(\[.*\])', decoded_label)[0])
            decoded_label.sort()
            decoded_label = ' '.join(decoded_label)
            try:
                find_res = re.findall(r'(\[.*\])', decoded_pred)
                tmp = []
                for res in find_res:
                    tmp += eval(res)
                decoded_pred = tmp
                decoded_pred.sort()
                decoded_pred = ' '.join(decoded_pred)
            except:
                decoded_pred = ''
        if class_name not in cmrc_data:
            cmrc_data[class_name] = {}
            cmrc_data[class_name]['decoded_preds'] = []
            cmrc_data[class_name]['decoded_labels'] = []
        cmrc_data[class_name]['decoded_preds'].append(decoded_pred)
        cmrc_data[class_name]['decoded_labels'].append(decoded_label)
        cmrc_data['all']['decoded_preds'].append(decoded_pred)
        cmrc_data['all']['decoded_labels'].append(decoded_label)

    cmrc_res_dict = {}
    for class_name in tqdm(cmrc_data):
        cmrc_res_dict[class_name] = compute_metrics(cmrc_data[class_name]['decoded_preds'],
                                                    cmrc_data[class_name]['decoded_labels'])

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(cmrc_res_dict, f, ensure_ascii=False)
    print('metric   all     code    table   game')
    for metric in ['rouge-l', 'bleu-4', 'recall']:
        print(f"{metric}: {cmrc_res_dict['all'][metric] * 100:.2f}"
              f"    {cmrc_res_dict['code'][metric] * 100:.2f}"
              f"    {cmrc_res_dict['table'][metric] * 100:.2f}"
              f"    {cmrc_res_dict['game'][metric] * 100:.2f}")


def batch_run(llm, tokenizer, max_length, data_path, save_path):
    if osp.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            pred_dict = json.load(f)
    else:
        pred_dict = {}

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    k = 0
    for line in tqdm(data):
        k += 1
        id = str(line['id'])
        if id in pred_dict and pred_dict[id]['answer'] != '':
            continue

        if 'llama2' in save_path:
            context = '''<s>[INST] <<SYS>>
{instruction} PLease answer as short as possible! <</SYS>>

{input} [/INST]'''.format(instruction=line['instruction'], input=line['input'])
        else:
            context = line['instruction'] + '\n' + line['input']
        context, exceed_limit = limit_token_length(tokenizer, max_length, prompt_template='', ori_context=context,
                                                   question='')
        try:
            pred = llm(context)
        except Exception as e:
            print(e)
            continue
        line['answer'] = pred
        # print(f'instruction: {instruction}')
        # print(f'pred: {pred}')
        del line['id']
        pred_dict[id] = line
        if k % 100 == 0:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(pred_dict, f, ensure_ascii=False)
            print('save')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f, ensure_ascii=False)
        print('save')


if __name__ == '__main__':
    models = [
        # 'chatglm3-6b',
        # 'chatglm3-6b-wo_100k',
        # 'chatglm3-6b-code_196k',
        # 'chatglm3-6b-table_196k',
        # 'chatglm3-6b-game_196k',
        # 'chatglm3-6b-game2_196k',
        # 'chatglm3-6b-generate_196k',
        # 'llama2-7b',
        # 'llama2-7b-wo_100k',
        # 'llama2-7b-wo_98k/checkpoint-8000',
        # 'llama2-7b-code_196k',
        # 'llama2-7b-table_200k',
        # 'llama2-7b-table_196k',
        # 'llama2-7b-game_196k',
        # 'llama2-7b-game2_196k',
        # 'llama2-7b-generate_196k',
        # 'llama2-13b',
        # 'llama2-13b-wo_98k/checkpoint-9000',
        'llama2-13b-code_196k',
        # 'llama2-13b-table_196k',
        # 'llama2-13b-game_196k',
        # 'llama2-13b-game2_196k',
        # 'llama2-13b-generate_196k',
        # 'baichuan2-7b',
        # 'baichuan2-7b-wo_100k',
        # 'baichuan2-7b-code_200k',
        # 'baichuan2-7b-table_200k',
        # 'baichuan2-7b-game_196k',
        # 'baichuan2-7b-game2_196k',
        # 'baichuan2-7b-generate_196k',
        # 'baichuan2-13b',
        # 'baichuan2-13b-wo_100k',
        # 'baichuan2-13b-code_200k',
        # 'baichuan2-13b-table_200k',
        # 'baichuan2-13b-game_196k',
        # 'baichuan2-13b-game2_196k',
        # 'baichuan2-13b-generate_196k',
        # 'gpt-3.5',
    ]
    for model in models:
        data_path = 'data/test.json'
        save_path = f'result/{model}.json'
        save_folder = osp.split(save_path)[0]
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
        llm, tokenizer, max_length = load_llm(model)
        batch_run(llm, tokenizer, max_length, data_path, save_path)
        if osp.exists(save_path):
            evaluate(save_path, save_path.replace('.json', '_score.json'))
        print(model)
        print('*' * 100)
