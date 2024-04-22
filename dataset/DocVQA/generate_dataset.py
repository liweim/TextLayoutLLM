import sys

project_root = '../..'
sys.path.insert(0, project_root)

from parsers.text_layout_parser import parse_text, merge_row
from tqdm import tqdm
import os.path as osp
import os
from datasets import load_from_disk
import json
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

folder = 'data/'
msr_ocr_path = f"{folder}/ocr_results"
gt_path = f'{folder}/qa.json'


def prepare_msr_ocr():
    msr_gt_dict = {}
    msr_data = load_from_disk(msr_ocr_path)
    for data in tqdm(msr_data):
        same_boxes, same_recs = merge_row(data['boxes'], data['words'], merge_same_row=False)
        ocr = '\n'.join(same_recs)
        msr_gt_dict[data['questionId']] = {'image': data['image'], 'question': data['question'],
                                           'answers': data['answers'], 'ocr': ocr}
    with open(f'{folder}/val_msr.json', 'w', encoding='utf-8') as f:
        json.dump(msr_gt_dict, f)


def prepare_origin_ocr():
    gt = json.load(open(gt_path))['data']
    gt_strip, gt_layout = {}, {}
    for data in tqdm(gt):
        boxes, words = [], []
        path = data['image']
        path = folder + path
        ext = osp.splitext(path)[1]
        ocr_path = path.replace('documents', 'ocr_results').replace(ext, '.json')
        lines = json.load(open(ocr_path, encoding='utf-8'))['recognitionResults'][0]['lines']
        for line in lines:
            box, word = line['boundingBox'], line['text']
            x1, y1, x2, y2 = min([box[0], box[2], box[4], box[6]]), min([box[1], box[3], box[5], box[7]]), max(
                [box[0], box[2], box[4], box[6]]), max([box[1], box[3], box[5], box[7]])
            boxes.append([x1, y1, x2, y2])
            words.append(word)
        boxes, words = merge_row(boxes, words, merge_same_row=False)
        ocr_layout = parse_text(boxes, words, col_sep=' ', row_sep='\n')
        ocr_strip = ' '.join(words)
        gt_strip[data['questionId']] = {'image': data['image'], 'question': data['question'], 'ocr': ocr_strip, 'answers': data['answers']}
        gt_layout[data['questionId']] = {'image': data['image'], 'question': data['question'], 'ocr': ocr_layout, 'answers': data['answers']}
    with open(f'{folder}/test.json', 'w', encoding='utf-8') as f:
        json.dump(gt_strip, f)
    with open(f'{folder}/test_space.json', 'w', encoding='utf-8') as f:
        json.dump(gt_layout, f)


if __name__ == '__main__':
    prepare_origin_ocr()
