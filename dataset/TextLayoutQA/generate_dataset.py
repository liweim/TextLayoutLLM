import sys

project_root = '../..'
sys.path.insert(0, project_root)

import numpy as np
import re
import json
import random
import copy


def generate_list():
    product_path = 'data/product.txt'
    list_path = 'data/shopping_list.json'

    products = open(product_path, 'r', encoding='utf-8').read().split('\n')
    shopping_list = []
    for i in range(300):
        ls = []
        for j in range(random.randint(2, 4)):
            l = random.sample(products, k=random.randint(1, 5))
            ls.append(l)
        shopping_list.append(ls)
    with open(list_path, 'w', encoding='utf-8') as f:
        json.dump(shopping_list, f, ensure_ascii=False)


def product_list_ocr(symbols, product_list, col_sep=' ', row_sep='\n'):
    bboxes = []
    texts = []
    product_list = [[symbols[i]] + list for i, list in enumerate(product_list)]
    max_x, max_y = 0, 0
    for k, products in enumerate(product_list):
        max_length = max([len(d) for d in products])
        if k == 0:
            x, y = random.randint(0, 2), random.randint(0, 2)
            max_x, max_y = x + max_length, y + len(products)
        elif k == 1:
            x, y = max_x + random.randint(2, 4), random.randint(0, 2)
            max_y = max(max_y, y + len(products))
        elif k == 2:
            x, y = random.randint(0, 2), max_y + random.randint(2, 4)
            max_x = max(max_x, x + max_length)
        else:
            x, y = max_x + random.randint(2, 4), max_y + random.randint(2, 4)
        product_bbox = [[x, y + i, x + len(product), y + i + 1] for i, product in enumerate(products)]
        bboxes += product_bbox
        texts += products

    bboxes = np.array(bboxes)
    w, h = max(bboxes[:, 2]), max(bboxes[:, 3])
    txt_array = np.empty((h, w), dtype=str)
    txt_array[:] = col_sep
    for (x1, y1, x2, y2), text in zip(bboxes, texts):
        txt_array[y1, x1:x2] = list(text)

    # reduce empty
    for i, row in enumerate(txt_array):
        if np.all(row == col_sep):
            txt_array[i] = row_sep
    ocr = row_sep.join([''.join(row) for row in txt_array])
    ocr = re.sub(row_sep + '{3,}', row_sep + row_sep, ocr)
    ocr = re.sub(r'^' + row_sep + '+', '', ocr)
    ocr = re.sub(row_sep + r'+$', '', ocr)
    return ocr


def generate_shopping_list(col_sep, path, save_path):
    instruction = '''Here are {num} shopping lists ({symbol}) with different products:
{content}'''

    shopping_lists = json.load(open(path, encoding='utf-8'))

    rets = []
    symbols = ['A', 'B', 'C', 'D']
    directions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    id = 0
    for shopping_list in shopping_lists:
        num = len(shopping_list)
        content = product_list_ocr(symbols, shopping_list, col_sep=col_sep, row_sep='\n')
        ocr = instruction.format(num=num, symbol=', '.join(symbols[:num]), content=content)

        index = random.choice(range(num))
        question = f'What products do shopping list {symbols[index]} contain?'
        answer = shopping_list[index]
        ret_dict = {'id': id, 'ocr': ocr, 'shopping_list': shopping_list, 'question': question, 'answer': answer,
                    'type': 'easy'}
        rets.append(ret_dict)
        id += 1

        index = random.sample(range(num), k=2)
        question = f'What products do shopping list {symbols[index[0]]} and {symbols[index[1]]} contain in total?'
        answer = copy.deepcopy(shopping_list[index[0]])
        for item in shopping_list[index[1]]:
            if item not in answer:
                answer.append(item)
        ret_dict = {'id': id, 'ocr': ocr, 'shopping_list': shopping_list, 'question': question, 'answer': answer,
                    'type': 'easy'}
        rets.append(ret_dict)
        id += 1

        index = random.choice(range(num))
        question = f'What products do shopping list in the {directions[index]} corner contain?'
        answer = shopping_list[index]
        ret_dict = {'id': id, 'ocr': ocr, 'shopping_list': shopping_list, 'question': question, 'answer': answer,
                    'type': 'hard'}
        rets.append(ret_dict)
        id += 1
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(rets, f, ensure_ascii=False)


def generate_shopping_list_strip(path, save_path):
    data = json.load(open(path, encoding='utf-8'))
    for line in data:
        ocr = line['ocr'].replace('\n', ' ')
        while '  ' in ocr:
            ocr = ocr.replace('  ', ' ')
        line['ocr'] = ocr
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == '__main__':
    generate_shopping_list(' ', 'data/shopping_list.json', 'data/test_shopping_space.json')
    generate_shopping_list('\t', 'data/shopping_list.json', 'data/test_shopping_tab.json')
    generate_shopping_list('\u02C6', 'data/shopping_list.json', 'data/test_shopping_caron.json')
    generate_shopping_list('a', 'data/shopping_list.json', 'data/test_shopping_a.json')
    generate_shopping_list_strip('data/test_shopping_space.json', 'data/test_shopping.json')
