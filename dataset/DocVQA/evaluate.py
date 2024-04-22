from sklearn.metrics import accuracy_score
import pandas as pd
import os
import sys

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

    gt_dict = json.load(open(gt_path))

    k = 0
    for id in tqdm(gt_dict.keys()):
        k += 1
        if str(id) in pred_dict and pred_dict[str(id)]['pred'] != '':
            continue
        data = gt_dict[id]
        question = data['question']
        ocr = data['ocr']
        try:
            answer, exceed_limit = agent(ocr, question)
            print(answer)
        except Exception as e:
            print(e)
            continue
        pred_dict[id] = {'image': data['image'], 'question': question,
                         'answers': data['answers'],
                         'pred': answer, 'exceed_limit': exceed_limit}

        if k % 100 == 0:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(pred_dict, f)
                print('save')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f)
        print('save')


def eval_llm(llm, model_type, update_metric=False):
    annotation_path = 'result/annotation.xlsx'
    df = pd.read_excel(annotation_path)
    gts = list(df['gt'].values)
    if update_metric:
        scores = list(df[model_type].values)
    else:
        data = df[['id', 'question', 'gt_answer', 'pred_answer']].values
        scores = []
        for id, question, gt, pred in data:
            score = compute_llm_eval(llm, question, gt, pred)
            scores.append(score)
        df[model_type] = scores
        df.to_excel(annotation_path, index=None)
    acc = accuracy_score(gts, scores)
    print(acc)


def compare_models(result_path1, result_path2):
    with open(result_path1, 'r', encoding='utf-8') as f:
        result1 = json.load(f)

    with open(result_path2, 'r', encoding='utf-8') as f:
        result2 = json.load(f)

    ids = result1.keys()
    scores, scores_llm = [], []
    score2s, score2s_llm = [], []
    for id in tqdm(ids):
        score = result1[id]['score']
        scores.append(score)
        score_llm = result1[id]['score_llm']
        scores_llm.append(score_llm)

        score = result2[id]['score']
        score2s.append(score)
        score_llm = result2[id]['score_llm']
        score2s_llm.append(score_llm)
    score_all = np.mean(scores)
    score_llm_all = np.mean(scores_llm)
    print(f'anls1: {score_all}, llm1: {score_llm_all}')

    score2_all = np.mean(score2s)
    score2_llm_all = np.mean(score2s_llm)
    print(f'anls2: {score2_all}, llm2: {score2_llm_all}')


if __name__ == '__main__':
    instruction = """Given the context: 
{context}

Use few words to answer.
Question: {question}
Answer: """

    for model in ['chatglm3-6b', 'llama2-7b', 'llama2-13b', 'baichuan2-7b', 'baichuan2-13b', 'gpt-3.5']:
        inst = instruction
        prompt = ''
        qa = DocQA(model, instruction=inst, show_log=False)
        for prefix in ['', '_space']:
            result_path = f'result/{model}{prefix}{prompt}.json'
            gt_path = f'data/test{prefix}.json'
            if model == 'gpt-3.5':
                with get_openai_callback() as cb:
                    run(qa, gt_path, result_path)
                print(cb)
            else:
                run(qa, gt_path, result_path)
            rephrase_result(qa.llm, result_path, update=True)
            if osp.exists(result_path):
                print(result_path)
                compute_metric(result_path, update=True, type='rephrase')
                print('*' * 100)
