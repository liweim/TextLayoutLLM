import sys
import os

project_root = '../..'
sys.path.insert(0, project_root)

from utils.metric import *
from model.docQA import DocQA

def run(agent, gt_path, result_path):
    if not osp.exists('result'):
        os.makedirs('result')
    if osp.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            pred_dict = json.load(f)
    else:
        pred_dict = {}

    gt = json.load(open(gt_path, encoding='utf-8'))

    k = 0
    for data in tqdm(gt):
        k += 1
        id = data['id']
        if str(id) in pred_dict and pred_dict[str(id)]['pred'] != '':
            continue
        ocr, question, gt_answer = data['ocr'], data['question'], data['answer']
        try:
            answer, exceed_limit = agent(ocr, question)
            print(answer)
        except Exception as e:
            print(e)
            continue
        pred_dict[id] = {'question': question, 'gt': gt_answer, 'pred': answer, 'exceed_limit': exceed_limit}
        if k % 100 == 0:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(pred_dict, f, ensure_ascii=False)
                print('save')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f, ensure_ascii=False)
        print('save')


def compute_metric(result_path, update=False, type='origin'):
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    ids = result.keys()
    scores, scores_limit = [], []
    bleu_scores, bleu_scores_limit = [], []

    if type == 'origin':
        for id in tqdm(ids):
            gt = result[id]['gt']
            pred = result[id]['pred']
            exceed_limit = result[id]['exceed_limit']

            if 'score' in result[id] and update == False:
                score = result[id]['score']
                bleu_score = result[id]['bleu_score']
            else:
                find_res = re.findall(r'Answer:(.*)', pred, re.IGNORECASE)
                if len(find_res) > 0:
                    pred = find_res[0].strip()
                rouge_score = compute_rouge(gt, pred)
                score = rouge_score["rouge-l"]['f']
                result[id]['score'] = score
                bleu_score = compute_bleu([gt], pred)
                result[id]['bleu_score'] = bleu_score
            scores.append(score)
            bleu_scores.append(bleu_score)
            if not exceed_limit:
                scores_limit.append(score)
                bleu_scores_limit.append(bleu_score)
        print(f'rouge-l: {np.mean(scores):.3%} ({np.mean(scores_limit):.3%})')
        print(f'bleu-4: {np.mean(bleu_scores):.3%} ({np.mean(bleu_scores_limit):.3%})')
    else:
        for id in tqdm(ids):
            gt = result[id]['gt']
            pred = result[id]['pred_rephrase']
            exceed_limit = result[id]['exceed_limit']

            if 'score_rephrase' in result[id] and update == False:
                score = result[id]['score_rephrase']
                bleu_score = result[id]['bleu_score_rephrase']
            else:
                find_res = re.findall(r'Answer:(.*)', pred, re.IGNORECASE)
                if len(find_res) > 0:
                    pred = find_res[0].strip()
                rouge_score = compute_rouge(gt, pred)
                score = rouge_score["rouge-l"]['f']
                result[id]['score_rephrase'] = score
                bleu_score = compute_bleu([gt], pred)
                result[id]['bleu_score_rephrase'] = bleu_score
            scores.append(score)
            bleu_scores.append(bleu_score)
            if not exceed_limit:
                scores_limit.append(score)
                bleu_scores_limit.append(bleu_score)
        print(f'rouge-l_rephrase: {np.mean(scores):.3%} ({np.mean(scores_limit):.3%})')
        print(f'bleu-4_rephrase: {np.mean(bleu_scores):.3%} ({np.mean(bleu_scores_limit):.3%})')

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
    instruction = """Given a table: 
{context}

Answer questions about the table.
Note: think step by step.
Question: {question}
Answer: """

    for model in ['chatglm3-6b', 'llama2-7b', 'llama2-13b', 'baichuan2-7b', 'baichuan2-13b', 'gpt-3.5']:
        inst = instruction
        prompt = ''
        qa = DocQA(model, instruction=inst, show_log=False)
        if model != 'baichuan2-13b-base':
            for prefix in ['_array', '_linear', '_triplet', '_space']:
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