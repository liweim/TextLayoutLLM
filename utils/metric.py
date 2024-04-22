# coding: utf-8
import re
from tqdm import tqdm
import json
import numpy as np
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import requests
from langchain.callbacks import get_openai_callback
from nltk import word_tokenize
import os.path as osp


def compute_accuracy(pred, gt, sep='a'):
    if type(gt) == str:
        gt = [gt]
    for g in gt:
        if g in pred and sep + g not in pred and g + sep not in pred:
            return 1
    return 0


def compute_perplexity(model, tokenizer, text, answer):
    import torch

    encodings = tokenizer(text, return_tensors="pt")
    answer_encodings = tokenizer(answer, return_tensors="pt")

    max_length = 4096
    device = model.device
    seq_len = encodings.input_ids.size(1)
    answer_seq_len = answer_encodings.input_ids.size(1)

    end_loc = min(max_length, seq_len)
    input_ids = encodings.input_ids[:, :end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-answer_seq_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    ppl = torch.exp(neg_log_likelihood)
    return ppl


def compute_anls(pred, gts, tau=0.5, rank=0):
    import textdistance as td

    max_s = 0
    for gt in gts:
        dis = td.levenshtein.distance(pred.lower(), gt.lower())
        max_len = max(len(pred), len(gt))
        if max_len == 0:
            s = 0
        else:
            nl = dis / max_len
            s = 1 - nl if nl < tau else 0
        max_s = max(s, max_s)
    return max_s


def compute_anls_improve(pred, gts, tau=0.5):
    # 平均归一化Levenshtein相似度
    import textdistance as td

    max_s = 0
    pred = pred.lower()
    for gt in gts:
        gt = gt.lower()

        if gt in pred:
            max_s = 1
            break
        else:
            dis = td.levenshtein.distance(pred, gt)
            max_len = max(len(pred), len(gt))
            if max_len == 0:
                s = 0
            else:
                nl = dis / max_len
                s = 1 - nl if nl < tau else 0
            max_s = max(s, max_s)
    return max_s


def tokenize(text, language):
    if language == 'en':
        tokens = word_tokenize(text)
    elif language == 'zh':
        tokens = list(jieba.cut(text))
    else:
        raise Exception('Only support English and Chinese!')
    return tokens


def is_contains_chinese(strs):
    count = 0
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            count += 1
    if count > 2:
        return True
    else:
        return False


def compute_recall(label, pred):
    if is_contains_chinese(label):
        language = 'zh'
    else:
        language = 'en'
    label = tokenize(label, language)
    pred = tokenize(pred, language)
    num = len(label)
    correct = 0
    for l in label:
        if l in pred:
            correct += 1
    recall = correct / num
    return recall


def compute_bleu(gt_label, prediction):
    if is_contains_chinese(gt_label[0]):
        language = 'zh'
    else:
        language = 'en'
    reference = [tokenize(label, language) for label in gt_label]
    hypothesis = tokenize(prediction, language)
    score = sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method4)
    return score


def compute_rouge(gt_label, prediction):
    if is_contains_chinese(gt_label):
        language = 'zh'
    else:
        language = 'en'

    if language == 'en':
        from rouge import Rouge
    elif language == 'zh':
        from rouge_chinese import Rouge
    else:
        raise Exception('Only support English and Chinese!')
    rouge = Rouge()

    if prediction == '' or gt_label == '' or len(tokenize(prediction, language)) == 0:
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        reference = " ".join(tokenize(gt_label, language))
        hypothesis = " ".join(tokenize(prediction, language))
        rouge_score = rouge.get_scores(hypothesis, reference)[0]
        return rouge_score


def compute_llm_eval(llm, question, gt, pred):
    prompt_template = """
        Question (A): 
        {question}
        
        Answer (B) to the question A: 
        {gt}
        
        Another answer (C) to the question A: 
        {pred}
        
        Does the answer C say the same thing as the answer B? Answer "yes" or "no"!
        """
    gt = str(gt)
    pred = str(pred)
    if gt.lower().replace(',', '').replace('.', '') in pred.lower().replace(',', '').replace('.', ''):
        eval = 1
        print('contain')
    else:
        prompt = prompt_template.format(question=question, gt=gt, pred=pred)
        eval = llm(prompt).lower().strip()
        print(eval)
        if 'no' in eval:
            eval = 0
        else:
            eval = 1
    return eval


def compute_metric(result_path, update=False, type='origin'):
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    ids = result.keys()
    scores, scores_llm = [], []

    if type == 'origin':
        for id in tqdm(ids):
            if 'score' in result[id] and update == False:
                score = result[id]['score']
            else:
                gt_label = result[id]['answers']
                pred = result[id]['pred']
                score = compute_anls_improve(pred, gt_label)
                result[id]['score'] = score
            scores.append(score)
        score_all = np.mean(scores)
        print(f'anls: {score_all:.3%}')
    else:
        for id in tqdm(ids):
            if 'score_rephrase' in result[id] and update == False:
                score = result[id]['score_rephrase']
            else:
                gt_label = result[id]['answers']
                pred = result[id]['pred_rephrase']
                if 'Here are the rephrased answers:' in pred:
                    pred = result[id]['pred']
                if ':' in pred:
                    pred = ':'.join(pred.split(':')[1:]).strip()
                result[id]['pred_rephrase'] = pred
                score = compute_anls(pred, gt_label)
                result[id]['score_rephrase'] = score
            scores.append(score)
        score_all = np.mean(scores)
        print(f'anls_rephrase: {score_all:.3%}')

    # for id in tqdm(ids):
    #     if 'score_llm' in result[id] and update == False:
    #         score_llm = result[id]['score_llm']
    #     else:
    #         score_llm = compute_llm_eval(llm, result[id]['question'], result[id]['answers'][0], result[id]['pred'])
    #         result[id]['score_llm'] = score_llm
    #     scores_llm.append(score_llm)
    # score_llm_all = np.mean(scores_llm)
    # print(f'llm: {score_llm_all}')

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


def find_lcs(s1, s2):
    """find the longest common subsequence between s1 ans s2"""
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    max_len = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > max_len:
                    max_len = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - max_len:p], max_len


def ngram(decoded_preds, decoded_labels):
    score_dict = {"accuracy": [], "rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": [], "f1": [],
                  "precision": [], "recall": []}

    no_std_ans_cnt, pred_lens, label_lens = 0, 0.0, 0.0
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_lens += len(pred)
        label_lens += len(label)
        # 开放式问答中label为空的时候：跳过
        if label == '':
            no_std_ans_cnt += 1
            continue
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))

        lcs, lcs_len = find_lcs(reference, hypothesis)

        if len(hypothesis) == 0:
            score_dict["precision"].append(0)
            score_dict["recall"].append(0)
            score_dict["f1"].append(0)
        else:
            precision = 1.0 * lcs_len / len(hypothesis)
            recall = 1.0 * lcs_len / len(reference)
            score_dict["precision"].append(precision)
            score_dict["recall"].append(recall)
            if recall == 0 and precision == 0:
                score_dict["f1"].append(0)
            else:
                score_dict["f1"].append((2 * precision * recall) / (precision + recall))

        if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
            result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
        else:
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
            result = scores[0]

        for k, v in result.items():
            score_dict[k].append(v["f"])

        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(bleu_score)
        score_dict["accuracy"].append(float(len(label) != 0 and pred[:len(label)] == label))

    res_dict = {k: round(float(np.mean(v)), 4) for k, v in score_dict.items()}
    res_dict['samples'] = len(decoded_labels)
    res_dict['no_std_ans_cnt'] = no_std_ans_cnt
    return res_dict


def compute_ppl(agent, gt_path, instruction, result_path, update=False):
    if osp.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            pred_dict = json.load(f)
    else:
        pred_dict = {}

    gt = json.load(open(gt_path, encoding='utf-8'))
    strip_gt = json.load(open(gt_path.replace('_space', ''), encoding='utf-8'))

    strip_ppls, ppls = [], []
    k = 0
    for strip_data, data in tqdm(zip(strip_gt, gt)):
        k += 1
        strip_ocr = strip_data['ocr']
        id, ocr, question, answer = data['id'], data['ocr'], data['question'], data['answer']
        id = str(id)
        answer = str(answer)
        strip_text = instruction.format(context=strip_ocr, question=question) + ' ' + answer
        text = instruction.format(context=ocr, question=question) + ' ' + answer
        if id in pred_dict and update == False:
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

        if k % 100 == 0:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(pred_dict, f, ensure_ascii=False)
                print('save')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f, ensure_ascii=False)
        print('save')
    print(f'strip ppl: {np.median(strip_ppls)}, rich ppl: {np.median(ppls)}')


def rephrase_result(llm, result_path, type='shortest', update=False):
    if type == 'shortest':
        instruction = '''Given the question and answer pair, rephrase the answer to provide the most straightforward response to the question with few words in English.
Example 1: 
Question: What is the name of the person in the CC field?
Answer: The name of the person in the CC field is Jo Spach.
Rephrased answer: Jo Spach

Example 2: 
Question: What is the given document about?
Answer: The given document appears to be a summary of a evaluation survey conducted by Telmark in a particular monthly region in 2014. The survey aimed to evaluate the effectiveness of Telmark's promotional programs in the region. The document provides information on various aspects of the survey, including the number of stores that received promotional materials, the percentage of stores that placed the materials in a visible location, and the number of stores that participated in the promotion. Additionally, the document includes information on the wholesale accounts sold by Telmark in the region and the percentage of accounts that refused the promotion.
Rephrased answer: region monthly telmark program evaluation survey

Example 3: 
Question: What is the % of Employees in 2012 based on graph 'Distribution of Value-Added'?
Answer: Based on the graph 'Distribution of Value-Added', it can be observed that the percentage of employees in 2012 is around 80%.
Rephrased answer: 80%

Now rephrase the answer based on the QA pair:
Question: {question}
Answer: {answer}
Rephrased answer: '''
    else:
        instruction = '''Given the question and answer pair below, rephrase the answer to provide the most straightforward response to the question in one sentence without explanation.
Question: {question}
Answer: {answer}'''

    if 'llama2' in result_path:
        tmp = 'Now rephrase the answer based on the QA pair:\nQuestion: {question}\nAnswer: {answer}\nRephrased answer: '
        instruction = "<s>[INST] <<SYS>>\n" + instruction.replace(tmp, '') + "<</SYS>>\n\n" + tmp + "[/INST]"
    print(instruction)

    with open(result_path, 'r', encoding='utf-8') as f:
        pred_dict = json.load(f)
        k = 0
    for key in tqdm(pred_dict.keys()):
        if 'pred_rephrase' in pred_dict[key] and update == False:
            pred = pred_dict[key]['pred_rephrase'].replace('₹', '')
            if pred.startswith('t '):
                pred = pred[2:]
            res = re.findall(r'(.*)\n\n', pred)
            if len(res) > 0:
                pred = res[0]
            pred_dict[key]['pred_rephrase'] = pred
            continue

        k += 1
        answer = pred_dict[key]['pred']
        if answer == '':
            rephrase_answer = ''
        else:
            question = pred_dict[key]['question']
            prompt = instruction.format(question=question, answer=answer)
            try:
                rephrase_answer = llm(prompt)
                if rephrase_answer != '':
                    if rephrase_answer[0] == '"' and rephrase_answer[-1] == '"':
                        rephrase_answer = rephrase_answer[1:-1]
                    if 'llama2' in result_path:
                        rephrase_answer = rephrase_answer.replace('₹', '')
                        if rephrase_answer.startswith('t '):
                            rephrase_answer = rephrase_answer[2:]
                        res = re.findall(r'(.*)\n\n', rephrase_answer)
                        if len(res) > 0:
                            rephrase_answer = res[0]
            except Exception as e:
                print(e)
                rephrase_answer = ''
            # print('question: ' + question)
            print('before: ' + answer)
            print('after: ' + rephrase_answer)
            print('*' * 100)
        pred_dict[key]['pred_rephrase'] = rephrase_answer
        if k % 100 == 0:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(pred_dict, f)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f)
