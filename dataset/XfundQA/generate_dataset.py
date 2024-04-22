import sys

project_root = '../..'
sys.path.insert(0, project_root)

from parsers.text_layout_parser import parse_text, merge_row
import json
from tqdm import tqdm

filter_ids = [4, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 183, 335, 336, 337,
              338, 339, 341, 360, 361, 376, 377, 379, 403, 406, 411, 446,
              476, 482, 491, 547, 555, 558, 561, 562, 563, 584, 586, 587,
              615, 619, 626, 638, 640, 643, 648, 658, 660, 661, 688, 689,
              696, 705, 720, 731, 733, 734, 735, 736, 737, 743, 744, 747,
              749, 751, 752, 754, 756, 757, 758, 759, 760, 761, 769, 776,
              785, 786, 801, 802, 818, 835, 855, 856, 859, 860, 862, 864,
              872, 873, 876, 879, 880, 883, 884, 885, 889, 907, 922, 934,
              937, 938, 943, 945, 948, 951, 981, 984, 988, 1010, 1016, 1022,
              1026, 1046, 1093, 1098, 1101, 1120, 1124, 1133, 1139, 1146, 1154]


def preprocess(a):
    flag = False
    if len(a) > 0:
        if a[0] == '(':
            a = a[1:]
        if a[-1] == ')':
            a = a[:-1]
        if '☑' in a and a != '☑':
            tmp = a[a.index('☑') + 1:]
            if tmp == '':
                a = a[:a.index('☑')]
            else:
                a = tmp
            if '□' in a:
                a = a[:a.index('□')]
            flag = True
        if '(√)' in a and a != '(√)':
            tmp = a[a.index('(√)') + 1:]
            if tmp == '':
                a = a[:a.index('(√)')]
            else:
                a = tmp
            if '()' in a:
                a = a[:a.index('()')]
            flag = True
    return a, flag


def refactor_ocr():
    path = 'data/zh.val.json'
    documents = json.load(open(path, encoding='utf-8'))['documents']
    strip_rets, layout_rets = [], []
    n = 0
    for document in tqdm(documents):
        boxes, words, link, id_text, id_label, qa = [], [], {}, {}, {}, {}
        lines = document['document']
        for line in lines:
            box, word, label, id, linking = line['box'], line['text'], line['label'], line['id'], line['linking']
            boxes.append(box)
            words.append(word)
            id_text[id] = word
            id_label[id] = label
        for line in lines:
            linking = line['linking']
            for k, v in linking:
                if id_label[k] == 'question' and id_text[k] not in qa and id_text[k] != '':
                    qa[id_text[k]] = []
                if id_label[v] in ['question', 'answer'] and id_text[k] in qa and id_text[v] not in qa[id_text[k]]:
                    qa[id_text[k]].append(id_text[v])
        ocr_strip = ' '.join(words)
        boxes, words = merge_row(boxes, words, merge_same_row=True)
        ocr_layout = parse_text(boxes, words, col_sep=' ', row_sep='\n')

        qas = []
        for question in qa.keys():
            if n not in filter_ids:
                answer = []
                for a in qa[question]:
                    a, flag = preprocess(a)
                    if flag:
                        answer = [a]
                        break
                    answer.append(a)
                qas.append({'id': n, 'question': question, 'answer': answer})
            n += 1
        ret_dict = {'image_id': document['id'], 'ocr': ocr_strip, 'qa': qas}
        strip_rets.append(ret_dict)
        ret_dict = {'image_id': document['id'], 'ocr': ocr_layout, 'qa': qas}
        layout_rets.append(ret_dict)
    with open('data/test.json', 'w', encoding='utf-8') as f:
        json.dump(strip_rets, f, ensure_ascii=False)
    with open('data/test_space.json', 'w', encoding='utf-8') as f:
        json.dump(layout_rets, f, ensure_ascii=False)


if __name__ == '__main__':
    refactor_ocr()
