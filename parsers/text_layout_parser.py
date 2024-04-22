import numpy as np
import re


def merge_row(boxes, words, merge_same_row=True, sep=' ', margin=3):
    def merge_module(same_row, font_size, same_boxes, same_recs, merge_same_row):
        def submodule(merge, boxes, recs):
            txt = sep.join([info[1] for info in merge])
            if len(merge) < 2:
                x1, y1, x2, y2 = merge[0][0]
                box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            else:
                merge_boxes = np.array([info[0] for info in merge]).reshape((-1, 4))
                x1, y1, x2, y2 = np.min(merge_boxes[:, 0]), np.min(merge_boxes[:, 1]), np.max(
                    merge_boxes[:, 2]), np.max(merge_boxes[:, 3])
                box = [x1, y1, x2, y2]
            boxes.append(box)
            recs.append(txt)

        same_row = sorted(same_row, key=lambda x: x[0][0])
        if merge_same_row:
            submodule(same_row, same_boxes, same_recs)
        else:
            merge = [same_row[0]]
            for i in range(len(same_row) - 1):
                if same_row[i + 1][0][0] - same_row[i][0][2] < margin * font_size:
                    merge.append(same_row[i + 1])
                else:
                    submodule(merge, same_boxes, same_recs)
                    merge = [same_row[i + 1]]
            submodule(merge, same_boxes, same_recs)

    font_sizes = [box[3] - box[1] for box in boxes]
    font_size = np.median(font_sizes) / 2

    same_recs = []
    same_boxes = []
    same_row = []
    last_y = boxes[0][1]
    for box, word in zip(boxes, words):
        if abs(box[1] - last_y) < font_size:
            box[1] = last_y
            same_row.append([box, word])
        else:
            last_y = box[1]
            if len(same_row) > 0:
                merge_module(same_row, font_size, same_boxes, same_recs, merge_same_row)
            same_row = [[box, word]]
    merge_module(same_row, font_size, same_boxes, same_recs, merge_same_row)
    return same_boxes, same_recs


def parse_text(boxes, texts, col_sep=' ', row_sep='\n', margin=0.7):
    x_sizes, y_sizes, min_xs = [], [], []
    for box, text in zip(boxes, texts):
        if len(text) > 0:
            x_size = (box[2] - box[0]) / len(text)
            y_size = box[3] - box[1]
            x_sizes.append(x_size)
            y_sizes.append(y_size)
    if len(x_sizes) == 0:
        return ''

    min_x_size = np.median(x_sizes) * margin
    min_y_size = np.median(y_sizes) * margin
    x_sizes = np.array(x_sizes)
    y_sizes = np.array(y_sizes)
    min_x_size = min(x_sizes[x_sizes > min_x_size])
    min_y_size = min(y_sizes[y_sizes > min_y_size])

    w = max(np.array(boxes)[:, 2])
    h = max(np.array(boxes)[:, 3])

    w = round(w / min_x_size) + 2
    h = round(h / min_y_size) + 2
    # print(min_x_size, min_y_size, w, h)
    txt_array = np.empty((h, w), dtype=str)
    txt_array[:] = col_sep

    for i in range(len(boxes)):
        if boxes[i][0] < min_x_size or boxes[i][1] < min_y_size or boxes[i][2] < min_x_size or boxes[i][
            3] < min_y_size:
            continue
        x1, y1, x2, y2 = np.int32(
            np.round([boxes[i][0] / min_x_size, boxes[i][1] / min_y_size, boxes[i][2] / min_x_size,
                      boxes[i][3] / min_y_size]))
        min_xs.append(x1)
        text = texts[i]
        word_list = np.array(list(text))
        if x2 - x1 > y2 - y1:
            if x1 + len(word_list) < w:
                txt_array[y1, x1:x1 + len(word_list)] = word_list
        else:
            if y1 + len(word_list) < y2:
                txt_array[y1:y1 + len(word_list), x1] = word_list

    if len(min_xs) > 0:
        min_x = min(min_xs)
        txt_array = txt_array[:, min_x:]

    # reduce empty
    row_mark = []
    for i, row in enumerate(txt_array):
        if np.all(row == col_sep):
            row_mark.append(i)

    mark = []
    txt_array = np.array(txt_array)
    num_col = txt_array.shape[1]
    for i in range(num_col):
        col = txt_array[:, i:min(num_col, i + 3)]
        if np.all(col == col_sep):
            mark.append(i)
    txt_array[:, mark] = ''
    reduce_txt_list = np.array([''.join(row) for row in txt_array])
    reduce_txt_list[row_mark] = row_sep

    ocr = row_sep.join(reduce_txt_list)
    ocr = re.sub(row_sep + '{3,}', row_sep + row_sep, ocr)
    ocr = re.sub(r'^' + row_sep + '+', '', ocr)
    ocr = re.sub(row_sep + r'+$', '', ocr)
    return ocr
