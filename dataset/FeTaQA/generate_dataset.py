import sys

project_root = '../..'
sys.path.insert(0, project_root)

from parsers.table_layout_parser import parse_table, linearize_table, triplet_table
import json
from tqdm import tqdm
import numpy as np


def refactor_ocr(path, save_path, type):
    data = open(path, encoding='utf-8').readlines()
    rets = []
    for line in tqdm(data):
        line = json.loads(line)
        id = line['feta_id']
        table_array = line['table_array']
        question = line['question']
        answer = line['answer']
        ocr = str(table_array)
        if type == 'linear':
            ocr = linearize_table(table_array)
        elif type == 'triplet':
            ocr = triplet_table(table_array)
        elif type == 'space':
            table_array = np.array(table_array)
            ocr = parse_table(table_array, ' ')
        ret_dict = {'id': id, 'ocr': ocr, 'question': question, 'answer': answer}
        rets.append(ret_dict)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(rets, f, ensure_ascii=False)


if __name__ == '__main__':
    path = 'data/fetaQA-v1_test.txt'
    save_path = 'data/test_array.json'
    refactor_ocr(path, save_path, type='array')
    save_path = 'data/test_linear.json'
    refactor_ocr(path, save_path, type='linear')
    save_path = 'data/test_triplet.json'
    refactor_ocr(path, save_path, type='triplet')
    save_path = 'data/test_space.json'
    refactor_ocr(path, save_path, type='space')
