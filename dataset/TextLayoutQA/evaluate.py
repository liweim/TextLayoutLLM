import sys
import os

project_root = '../..'
sys.path.insert(0, project_root)

from utils.metric import *
from model.docQA import DocQA


def run(agent, gt_path, result_path, update=False, num=None):
    if not osp.exists('result'):
        os.makedirs('result')
    if osp.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            pred_dict = json.load(f)
    else:
        pred_dict = {}

    gt = json.load(open(gt_path, encoding='utf-8'))

    k = 0
    if num is None:
        num = len(gt)
    for data in tqdm(gt[:num]):
        id = str(data['id'])
        if not update and id in pred_dict and pred_dict[id]['pred'] != '' and 'success' in pred_dict[id] and \
                pred_dict[id]['success'] == True:
            continue
        k += 1
        ocr, question, gt_answer, type = data['ocr'], data['question'], data['answer'], data['type']
        fail = True
        count = 0
        while fail and count < 1:
            answer, exceed_limit = agent(ocr, question)
            # print(answer)
            if '[' in answer and ']' in answer:
                fail = False
            else:
                count += 1

        pred_dict[id] = {'question': question, 'gt': gt_answer, 'pred': answer, 'exceed_limit': exceed_limit,
                         'type': type}
        if k % 100 == 0:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(pred_dict, f, ensure_ascii=False)
                print('save')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f, ensure_ascii=False)
        print('save')


def compute_metric(result_path, update=False):
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    ids = result.keys()
    scores, scores_limit = [], []
    fail = 0
    for id in tqdm(ids):
        gt = result[id]['gt']
        pred = result[id]['pred']
        exceed_limit = result[id]['exceed_limit']

        if 'score' in result[id] and update == False:
            score = result[id]['score']
        else:
            find_res = re.findall(r'(\[[^]]*\])', pred)
            try:
                assert len(find_res) > 0
                tmp = []
                for res in find_res:
                    tmp += list(eval(res))
                pred = tmp
                result[id]['success'] = True
            except:
                if len(find_res) > 0:
                    pred = ''.join(find_res)
                pred = re.findall(r'\b\w+\b', pred)
                fail += 1
                result[id]['success'] = False
            pred = list(set(pred))

            correct = 0
            for g in gt:
                if g in pred:
                    correct += 1
            if correct == 0:
                score = 0
            else:
                precision = correct / len(pred)
                recall = correct / len(gt)
                score = 2 * precision * recall / (precision + recall)
            result[id]['score'] = score
        scores.append(score)
        if not exceed_limit:
            scores_limit.append(score)
    print(f'F1: {np.mean(scores):.3%} ({np.mean(scores_limit):.3%}), fail: {fail}')

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
    instruction = '''Given some shopping lists with different products, you are supposed to enumerate the products of specific lists 
and answer questions in the form of a list, for example: ['a', 'b'], reply with the list only! If you don't know the answer, reply with the empty list [].

For example:
Here are 2 shopping lists (A, B) with different products:
A       B
apple   fish
banana  chair
car

C
dog
cat

Question: What products do shopping list B and C contain in total?
Answer: ['fish', 'chair', 'dog', 'cat']

Now answer the question below:
{context}

Question: {question}
Answer: '''

    instruction_llama = '''<s>[INST] <<SYS>>
Given some shopping lists with different products, you are supposed to enumerate the products of specific lists
and answer questions in the form of a list, for example: ['a', 'b'], reply with the list only! If you don't know the answer, reply with the empty list [].

For example:
Here are 2 shopping lists (A, B) with different products:
A       B
apple   fish
banana  chair
car

C
dog
cat

Question: What products do shopping list B and C contain in total?
Answer: ['fish', 'chair', 'dog', 'cat']

<</SYS>>
Now answer the question below:
{context}

{question} [/INST]'''

    models = [
        # 'chatglm3-6b-base',
        'chatglm3-6b',
        # 'chatglm3-6b-wo',
        # 'chatglm3-6b-code',
        # 'chatglm3-6b-table',
        # 'chatglm3-6b-generate',
        # 'llama2-7b-base',
        # 'llama2-7b',
        # 'llama2-7b-wo',
        # 'llama2-7b-code',
        # 'llama2-7b-table',
        # 'llama2-7b-generate',
        # 'llama2-13b-base',
        # 'llama2-13b',
        # 'llama2-13b-wo',
        # 'llama2-13b-code',
        # 'llama2-13b-table',
        # 'llama2-13b-generate',
        # 'baichuan2-7b-base',
        # 'baichuan2-7b',
        # 'baichuan2-7b-wo',
        # 'baichuan2-7b-code',
        # 'baichuan2-7b-table',
        # 'baichuan2-7b-generate',
        # 'baichuan2-13b-base'
        # 'baichuan2-13b',
        # 'baichuan2-13b-wo',
        # 'baichuan2-13b-code',
        # 'baichuan2-13b-table',
        # 'baichuan2-13b-generate',
        # 'gpt-3.5',
    ]
    for model in models:
        inst = instruction
        prompt = ''
        if model.startswith('llama2'):
            inst = instruction_llama
        if not model.endswith('-base'):
            qa = DocQA(model, instruction=inst, show_log=False, device='auto')
            for prefix in ['_space', '']:  # '', '_space', '_tab', '_caron', '_a'
                result_path = f'result/{model}{prefix}{prompt}.json'
                save_folder = osp.split(result_path)[0]
                if not osp.exists(save_folder):
                    os.makedirs(save_folder)
                gt_path = f'data/test_shopping{prefix}.json'
                run(qa, gt_path, result_path, update=False)
                if osp.exists(result_path):
                    print(result_path)
                    compute_metric(result_path, update=True)
                    print('*' * 100)
        if model in [
            'chatglm3-6b-base',
            'chatglm3-6b',
            'llama2-7b-base',
            'llama2-7b',
            'llama2-13b-base',
            'llama2-13b',
            'baichuan2-7b-base',
            'baichuan2-7b',
            'baichuan2-13b-base'
            'baichuan2-13b',
            ]:
            qa = DocQA(model, instruction=inst, show_log=False, device='cuda')
            result_path = f'result/{model}_ppl.json'
            gt_path = 'data/test_shopping_space.json'
            compute_ppl(qa, gt_path, instruction, result_path, update=True)
