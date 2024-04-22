import sys

project_root = '../..'
sys.path.insert(0, project_root)

from parsers.table_layout_parser import parse_table
import pandas as pd
import string
import numpy as np
import re
import json
import random
import jsonlines
import glob
import os.path as osp
from tqdm import tqdm
from pandas import read_parquet
import copy
from sklearn.model_selection import train_test_split


def generate_base(type):
    if type == 'train':
        root_path = 'data/instructions/train/'
        exclude_code_files = [root_path + path for path in ['CodeAlpaca/EN/code_alpaca.json', 'GPT4all/EN/gpt4all.json',
                                                            'GPT4all/EN/gpt4all_without_p3.json',
                                                            'GPTeacher/EN/codegen-instruct.json',
                                                            'COIG/CN/leetcode.json']]

        train_set = []
        for path in tqdm(glob.glob(root_path + '*/*/*.json')):
            if path in exclude_code_files:
                print(f'exclude path: {path}')
                continue
            print(path)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                continue
            source = osp.split(osp.split(osp.split(path)[0])[0])[1]
            for item in data:
                text = item['instruction'] + item['input'] + item['output']
                if 'def ' in text or 'public class ' in text or '   ' in text or '\t\t' in text:
                    continue
                item['source'] = source
                train_set.append(item)

        print(len(train_set))  # 12187375
        idx = list(range(len(train_set)))
        random.shuffle(idx)

        firefly_format = []
        for id in tqdm(idx[:100000]):
            data = train_set[id]
            sample = {}
            sample['category'] = data['source']
            sample['conversation'] = [{}]
            sample['conversation'][0]['human'] = data['instruction'] + '\n' + data['input']
            sample['conversation'][0]['assistant'] = data['output']
            firefly_format.append(sample)
        del train_set

        with jsonlines.open('data/train_wo_100k.jsonl', mode='w') as writer:
            for data in firefly_format:
                writer.write(data)

        split_dataset('data/train_wo_100k.jsonl', 'data/train_wo_98k.jsonl', 'data/eval_wo_2k.jsonl')
    else:
        sources = ['Generation', 'Answering', 'Classification', 'Rewriting', 'Mathematics']
        with open('data/test_base_all.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        res = {}
        for src in data.keys():
            source = src.split(' ')[-1]
            if source in sources:
                if source not in res:
                    res[source] = []
                res[source] += data[src]

        id = 0
        filter_data = []
        for src in res.keys():
            lines = res[src]
            random.shuffle(lines)
            for line in lines[:100]:
                line['id'] = id
                line['source'] = src.lower()
                id += 1
                filter_data.append(line)
        with open('data/test_base.json', 'w', encoding='utf-8') as f:
            json.dump(filter_data, f, ensure_ascii=False)


def generate_code(type):
    if type == 'train':
        with open('data/train_wo_100k.jsonl', 'r', encoding='utf-8') as f:
            base_data = [json.loads(jline) for jline in f.read().splitlines()]

        root_path = 'data/instructions/train/'
        paths = [root_path + path for path in ['CodeAlpaca/EN/code_alpaca.json', 'GPT4all/EN/gpt4all.json',
                                               'GPT4all/EN/gpt4all_without_p3.json',
                                               'GPTeacher/EN/codegen-instruct.json', 'COIG/CN/leetcode.json']]

        train_set = []
        for path in paths:
            print(path)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                continue
            source = osp.split(osp.split(osp.split(path)[0])[0])[1]
            for item in data[:35000]:
                item['source'] = source
                train_set.append(item)

        print(len(train_set))  # 106294
        idx = list(range(len(train_set)))
        random.shuffle(idx)

        firefly_format = []
        for id in tqdm(idx[:100000]):
            data = train_set[id]
            sample = {}
            sample['category'] = data['source']
            sample['conversation'] = [{}]
            sample['conversation'][0]['human'] = (data['instruction'] + data['input']).replace(' ', '\n').replace(' ',
                                                                                                                  '\n').replace(
                ' ', ' ')
            sample['conversation'][0]['assistant'] = data['output'].replace(' ', '\n').replace(' ', '\n').replace(' ',
                                                                                                                  ' ')
            firefly_format.append(sample)
        del train_set

        firefly_format += base_data

        with jsonlines.open('data/train_code_200k.jsonl', mode='w') as writer:
            for data in firefly_format:
                writer.write(data)

        split_dataset('data/train_code_200k.jsonl', 'data/train_code_196k.jsonl', 'data/eval_code_4k.jsonl')
    else:
        data = read_parquet("data/instructions/eval/oa_leet10k/oa_leet10k_contests_merged.parquet")

        eval_set = []
        for instruction, output, _ in data.values:
            sample = {
                "instruction": instruction.replace(' ', ' '),
                "input": '',
                "output": output.replace(' ', ' '),
                "source": 'code'
            }
            eval_set.append(sample)

        print(len(eval_set))  # 13973
        random.shuffle(eval_set)

        filter_data = []
        for i in range(100):
            sample = eval_set[i]
            sample['id'] = i
            filter_data.append(sample)

        with open('data/test_code.json', 'w', encoding='utf-8') as f:
            json.dump(filter_data, f, ensure_ascii=False)


def generate_table(type):
    if type == 'train':
        with open('data/train_wo_100k.jsonl', 'r', encoding='utf-8') as f:
            base_data = [json.loads(jline) for jline in f.read().splitlines()]

        train_set = []
        source = 'WikiTableQuestions'
        for path in tqdm(glob.glob('data/WikiTableQuestions/csv/*/*.csv')):
            try:
                df = pd.read_csv(path)
                idx_name = df.columns[0]
                table_array = df.values.astype(str)
                ocr = parse_table(table_array, ' ')
                if len(ocr) > 4000:
                    continue
                for idx in range(1, len(df)):
                    for col in df.columns[1:]:
                        sample = {}
                        sample['category'] = source
                        sample['conversation'] = [{}]
                        sample['conversation'][0][
                            'human'] = f'Given a table: \n{ocr}\nWhat is the {col} of {idx_name} {df.loc[idx, idx_name]}? ' \
                                       f'The answer is already shown in the table, use minimal words to answer.'
                        sample['conversation'][0][
                            'assistant'] = str(df.loc[idx, col])
                        train_set.append(sample)
            except:
                continue

        print(len(train_set))  # 121097
        idx = list(range(len(train_set)))
        random.shuffle(idx)

        firefly_format = []
        for id in idx[:100000]:
            firefly_format.append(train_set[id])
        del train_set

        firefly_format += base_data
        with jsonlines.open('data/train_table_200k.jsonl', mode='w') as writer:
            for data in firefly_format:
                writer.write(data)

        split_dataset('data/train_table_200k.jsonl', 'data/train_table_196k.jsonl', 'data/eval_table_4k.jsonl')
    else:
        eval_set = []
        data = open('data/fetaQA-v1_test.txt', encoding='utf-8').readlines()
        for line in tqdm(data):
            line = json.loads(line)
            table_array = line['table_array']
            table_array = np.array(table_array)
            columns = table_array[0]
            idx_name = columns[0]
            ocr = parse_table(table_array.copy(), ' ')
            if len(ocr) > 4000:
                continue
            for i in range(1, len(table_array)):
                for j in range(1, len(columns)):
                    col = columns[j]
                    sample = {
                        "instruction": f'Given a table: \n{ocr}\nWhat is the {col} of {idx_name} {table_array[i, 0]}? '
                                       f'The answer is already shown in the table, use minimal words to answer.',
                        "input": '',
                        "output": str(table_array[i, j]),
                        "source": 'table'
                    }
                    eval_set.append(sample)

        print(len(eval_set))  # 102053
        idx = list(range(len(eval_set)))
        random.shuffle(eval_set)

        filter_data = []
        for i, id in enumerate(idx[:100]):
            eval_set[id]['id'] = i
            filter_data.append(eval_set[id])

        with open('data/test_table.json', 'w', encoding='utf-8') as f:
            json.dump(filter_data, f, ensure_ascii=False)


def filter_sentences():
    import string
    punctuation_string = string.punctuation

    sentence_list = open('data/sentences_origin.txt', 'r').read().splitlines()
    with open('data/sentences.txt', 'w') as f:
        for sentence in sentence_list:
            for i in punctuation_string:
                if i == "'":
                    continue
                sentence = sentence.replace(i, '')
            words = sentence.split(' ')
            if len(words) > 1 and len(words) < 8:
                f.write(sentence + '\n')


def generate_sentence_search_puzzle(type):
    sentence_list = open('data/sentences.txt', 'r').read().splitlines()

    instruction_format = '''Sentence search puzzle is a game that involves a grid of words, where players are tasked with finding meaningful sentences hidden within the grid. 
The challenge lies in locating continuous words that make up meaningful sentences horizontally and vertically. 
The unused spaces in the grid are usually filled with random words to add complexity to the puzzle.
Note: answer in the form of a list, for example: ['a', 'b']. If you don't know the answer, reply with the empty list [].
Here is a toy example:
---------------------------
good  morning  i      dog  heat   get 
null  blue     eat    am   on     some
wood  oh       visit  dig  happy  food
---------------------------
First, search horizontally and find "good morning".
Then, search vertically and find "get some food".
So all the sentences hidden in this puzzle are: ["good morning", "get some food"].

Let's solve the following sentence search puzzle step by step:
---------------------------
{context}
---------------------------
'''
    answer_format = '''First, search horizontally and find {horizontal}.
Then, search vertically and find {vertical}.
So all the sentences hidden in this puzzle are: {answer}.'''

    firefly_format = []
    i = 0
    if type == 'train':
        total_num = 100000
    else:
        total_num = 100
    while i < total_num:
        num = random.randint(1, 5)
        sentences = random.sample(sentence_list, num)
        max_length = max([len(sentence.split(' ')) for sentence in sentences])
        size = random.randint(max_length + 1, max_length + 5)
        board, horizontal, vertical = sentence_search_puzzle(sentences, size)
        sentences = horizontal + vertical
        if len(sentences) == 0:
            continue
        if len(horizontal) == 0:
            horizontal = 'nothing'
        else:
            horizontal = ', '.join(horizontal)
        if len(vertical) == 0:
            vertical = 'nothing'
        else:
            vertical = ', '.join(vertical)
        # print(words, num, size)
        # print(board)

        instruction = instruction_format.format(context=board)
        answer = answer_format.format(horizontal=horizontal, vertical=vertical, answer=str(sentences))
        if type == 'train':
            sample = {}
            sample['category'] = 'game'
            sample['conversation'] = [{}]
            sample['conversation'][0]['human'] = instruction
            sample['conversation'][0]['assistant'] = answer
        else:
            sample = {
                "id": i,
                "instruction": instruction,
                "input": '',
                "output": answer,
                "source": 'game'
            }
        firefly_format.append(sample)
        i += 1

    if type == 'train':
        with open('data/train_wo_100k.jsonl', 'r', encoding='utf-8') as f:
            base_data = [json.loads(jline) for jline in f.read().splitlines()]

        with jsonlines.open('data/train_generate_200k.jsonl', mode='w') as writer:
            for data in firefly_format:
                writer.write(data)
            for data in base_data:
                writer.write(data)

        split_dataset('data/train_generate_200k.jsonl', 'data/train_generate_196k.jsonl', 'data/eval_generate_4k.jsonl')
    else:
        with open('data/test_generate.json', 'w', encoding='utf-8') as f:
            json.dump(firefly_format, f, ensure_ascii=False)


def sentence_search_puzzle(sentences, size):
    import random

    def place_sentence(board, sentence, horizontal, vertical, diagonal):
        # Randomly choose orientation: 0=horizontal, 1=vertical, 2=diagonal
        orientation = random.randint(0, 2)
        words = sentence.split(' ')
        sentence = f'"{sentence}"'

        placed = False
        count = 0
        while not placed and count < 5:
            count += 1
            if orientation == 0:  # Horizontal
                row = random.randint(0, len(board) - 1)
                col = random.randint(0, len(board) - len(words))
                # reverse = random.choice([True, False])
                # if reverse:
                #     words = words[::-1]
                space_available = all(board[row][c] == '-' or
                                      board[row][c] == words[i]
                                      for i, c in enumerate(range(col, col + len(words))))
                if space_available:
                    for i, c in enumerate(range(col, col + len(words))):
                        board[row][c] = words[i]
                    placed = True
                    horizontal.append(sentence)

            elif orientation == 1:  # Vertical
                row = random.randint(0, len(board) - len(words))
                col = random.randint(0, len(board) - 1)
                # reverse = random.choice([True, False])
                # if reverse:
                #     words = words[::-1]
                space_available = all(board[r][col] == '-' or
                                      board[r][col] == words[i]
                                      for i, r in enumerate(range(row, row + len(words))))
                if space_available:
                    for i, r in enumerate(range(row, row + len(words))):
                        board[r][col] = words[i]
                    placed = True
                    vertical.append(sentence)

            # elif orientation == 2:  # Diagonal top-left to bottom right
            #     row = random.randint(0, len(board) - len(words))
            #     col = random.randint(0, len(board) - len(words))
            #     # reverse = random.choice([True, False])
            #     # if reverse:
            #     #     words = words[::-1]
            #     space_available = all(board[r][c] == '-' or
            #                           board[r][c] == words[i]
            #                           for i, (r, c) in enumerate(zip(range(row, row + len(words)),
            #                                                          range(col, col + len(words)))))
            #     if space_available:
            #         for i, (r, c) in enumerate(zip(range(row, row + len(words)),
            #                                        range(col, col + len(words)))):
            #             board[r][c] = words[i]
            #         placed = True
            #         diagonal.append(sentence)
            #
            # elif orientation == 3:  # Diagonal bottom-left to top-right
            #     row = random.randint(len(words) - 1, len(board) - 1)
            #     col = random.randint(0, len(board) - len(words))
            #     # reverse = random.choice([True, False])
            #     # if reverse:
            #     #     words = words[::-1]
            #     space_available = all(board[r][c] == '-' or
            #                           board[r][c] == words[i]
            #                           for i, (r, c) in enumerate(zip(range(row, row - len(words), -1),
            #                                                          range(col, col + len(words)))))
            #     if space_available:
            #         for i, (r, c) in enumerate(zip(range(row, row - len(words), -1),
            #                                        range(col, col + len(words)))):
            #             board[r][c] = words[i]
            #         placed = True
            #         diagonal.append(sentence)

    def fill_empty(board):
        for row in range(len(board)):
            for col in range(len(board)):
                if board[row][col] == '-':
                    board[row][col] = random.choice(word_list)
                    # if random.random() < 0.5:
                    #     board[row][col] = ''

    def create_sentence_search(sentences, size):
        horizontal, vertical, diagonal = [], [], []
        board = [['-' for _ in range(size)] for _ in range(size)]

        for sentence in sentences:
            place_sentence(board, sentence.lower(), horizontal, vertical, diagonal)

        fill_empty(board)

        board_text = parse_table(np.array(board))

        return board_text, horizontal, vertical

    word_list = open('data/words.txt', 'r').read().splitlines()
    return create_sentence_search(sentences, size)


def merge_test():
    path = 'data/test_base.json'
    with open(path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    id = len(base_data)

    for p in ['data/test_code.json', 'data/test_table.json',
              'data/test_generate.json']:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for line in data:
            line['id'] = id
            id += 1
            base_data.append(line)

    with open('data/test.json', 'w', encoding='utf-8') as f:
        json.dump(base_data, f, ensure_ascii=False)
    print(id)


def split_dataset(path, train_path, eval_path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(jline) for jline in f.read().splitlines()]
    train, eval = train_test_split(data, test_size=0.02)
    with jsonlines.open(train_path, mode='w') as writer:
        for d in train:
            writer.write(d)
    with jsonlines.open(eval_path, mode='w') as writer:
        for d in eval:
            writer.write(d)


if __name__ == '__main__':
    for type in ['train', 'test']:
        generate_base(type=type)
        generate_code(type=type)
        generate_table(type=type)
        generate_sentence_search_puzzle(type=type)
    merge_test()
