import os
import sys

project_root = '../..'
sys.path.insert(0, project_root)

from utils.metric import *
from model.docQA import DocQA
from generate_dataset import preprocess


def run(agent, gt_path, result_path):
    if not osp.exists('result'):
        os.makedirs('result')
    if osp.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            pred_dict = json.load(f)
    else:
        pred_dict = {}

    gt = json.load(open(gt_path, encoding='utf-8'))

    for data in tqdm(gt):
        ocr, qas = data['ocr'], data['qa']
        for qa in qas:
            id, question, gt_answer = qa['id'], qa['question'], qa['answer']
            if str(id) in pred_dict and pred_dict[str(id)]['pred'] != '':
                continue

            try:
                answer, exceed_limit = agent(ocr, question)
            except Exception as e:
                print(e)
                continue
            pred_dict[id] = {'question': question, 'gt': gt_answer, 'pred': answer, 'exceed_limit': exceed_limit}
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(pred_dict, f, ensure_ascii=False)
            print('save')


def compute_metric(result_path, update=False):
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    ids = result.keys()
    num_gt_limit, correct_limit = 0, 0
    num_gt, correct = 0, 0

    for id in tqdm(ids):
        gt = result[id]['gt']
        pred = result[id]['pred']
        pred, flag = preprocess(pred)
        exceed_limit = result[id]['exceed_limit']

        if 'score' in result[id] and update == False:
            score = result[id]['score']
        else:
            score = 0
            for g in gt:
                score += compute_accuracy(pred, g)
            score /= len(gt)
            result[id]['score'] = score
        num_gt += len(gt)
        correct += int(score * len(gt))
        if not exceed_limit:
            num_gt_limit += len(gt)
            correct_limit += int(score * len(gt))

    num_pred = correct
    precision = correct / num_pred
    recall = correct / num_gt
    f1 = 2 * precision * recall / (precision + recall)
    num_pred_limit = correct_limit
    precision_limit = correct_limit / num_pred_limit
    recall_limit = correct_limit / num_gt_limit
    f1_limit = 2 * precision_limit * recall_limit / (precision_limit + recall_limit)
    print(f'f1: {f1:.3%} ({f1_limit:.3%})')

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


def compute_ppl(agent, gt_path, instruction, result_path):
    if osp.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            pred_dict = json.load(f)
    else:
        pred_dict = {}

    gt = json.load(open(gt_path, encoding='utf-8'))
    strip_gt = json.load(open(gt_path.replace('_space', ''), encoding='utf-8'))

    strip_ppls, ppls = [], []
    for strip_data, data in tqdm(zip(strip_gt, gt)):
        strip_ocr = strip_data['ocr']
        ocr, qas = data['ocr'], data['qa']
        for qa in qas:
            id, question, answer = qa['id'], qa['question'], qa['answer']
            id = str(id)

            answer = ''.join(answer)
            strip_text = instruction.format(context=strip_ocr, question=question) + ' ' + answer
            text = instruction.format(context=ocr, question=question) + ' ' + answer
            if id in pred_dict:
                strip_ppl = pred_dict[id]['strip_ppl']
                strip_ppls.append(strip_ppl)
                ppl = pred_dict[id]['ppl']
                ppls.append(ppl)
                continue

            strip_tokens = agent.tokenizer.encode(strip_text)
            tokens = agent.tokenizer.encode(text)
            if len(tokens) > agent.max_length:
                exceed_limit = True
                print('exceed_limit')
            else:
                exceed_limit = False
            gap = len(tokens) - len(strip_tokens)
            if gap > 0:
                strip_text = '\n' * gap + strip_text
            else:
                text = '\n' * -gap + text

            strip_ppl = compute_perplexity(agent.llm.model, agent.tokenizer, strip_text, answer)
            strip_ppl = float(strip_ppl)
            strip_ppls.append(strip_ppl)
            ppl = compute_perplexity(agent.llm.model, agent.tokenizer, text, answer)
            ppl = float(ppl)
            ppls.append(ppl)
            pred_dict[id] = {'strip_ppl': strip_ppl, 'ppl': ppl, 'exceed_limit': exceed_limit}

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(pred_dict, f, ensure_ascii=False)
            print('save')
    print(f'strip ppl: {np.median(strip_ppls)}, rich ppl: {np.median(ppls)}')


if __name__ == '__main__':
    instruction = """以下是一个由键值对组成的表单:"{context}"，请根据给定表单来回答。
提示：值一般出现在键的附近。仔细思考并用几个词来回答。
给定问题：请问键"{question}"的值是什么？
答案:"""

    for model in ['chatglm3-6b', 'llama2-7b', 'llama2-13b', 'baichuan2-7b', 'baichuan2-13b', 'gpt-3.5']:
        inst = instruction
        prompt = ''
        if model.startswith('llama2'):
            tmp = '给定问题：请问键"{question}"的值是什么？\n答案:'
            inst = '<s>[INST] <<SYS>>\n' + inst.replace(tmp, '') + '<</SYS>>\n\n请问键"{question}"的值是什么？ [/INST]'
            print(inst)
        qa = DocQA(model, instruction=inst, show_log=False, language='zh')
        for prefix in ['', '_space']:
            result_path = f'result/{model}{prefix}{prompt}.json'
            gt_path = f'data/test{prefix}.json'
            if model == 'gpt-3.5':
                with get_openai_callback() as cb:
                    run(qa, gt_path, result_path)
                print(cb)
            else:
                run(qa, gt_path, result_path)
            if osp.exists(result_path):
                compute_metric(result_path, update=True)
                print(result_path)
                print('*' * 100)
