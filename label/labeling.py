# labeling code

import csv

def labeling(path):
    # create label dictionary
    label_dict = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        rdr = csv.reader(f)
        for line in rdr:
            line = [x for x in line if x]
            label_dict[line[0]] = line[1:]

    f.close

    return label_dict