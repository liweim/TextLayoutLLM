def parse_table(table_array, col_sep=' ', row_sep='\n'):
    for i in range(table_array.shape[1]):
        col_width = max([len(d) for d in table_array[:, i]])
        for j in range(table_array.shape[0]):
            txt = table_array[j, i].replace('\n', ' ')
            table_array[j, i] = txt + col_sep * (col_width - len(txt))
    col_sep = col_sep * 2
    ocr = row_sep.join([col_sep.join(row) for row in table_array])
    return ocr


def linearize_table(table):
    ocr = " [ROW] ".join([str(i) + " " + " | ".join(row) for i, row in enumerate(table)])
    ocr = '[HEAD]' + ocr[1:]
    return ocr


def triplet_table(table):
    cols = table[0]
    lines = []
    for i, row in enumerate(table[1:]):
        for j, col in enumerate(row):
            lines.append(f'Row{i + 1} | {cols[j]} | {col}')
    ocr = '\n'.join(lines)
    return ocr
