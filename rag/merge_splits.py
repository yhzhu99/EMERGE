import json
import pandas as pd
import pdb

chunk_list = ['./output.json', './output3000.json', './output6000.json',
              './output9000.json', './output12000.json', './output15000.json',
              './output18000.json']

def merge():
    id2entities = {}
    for c in chunk_list:
        with open(c, encoding='utf8') as f:
            for line in f:
                line = line.rstrip()
                obj = json.loads(line)
                pid, entities, text = obj['PatientID'], obj['Entities'], str(obj['Texts'])
                entities = set(entities)
                true_entities = []
                for e in entities:
                    cnt = text.lower().count(str(e).lower())
                    if cnt > 0:
                        true_entities.append((e, cnt))
                true_entities.sort(key=lambda x: x[1], reverse=True)
                # pdb.set_trace()
                true_entities = [_[0] for _ in true_entities]  # 按出现频次降序排列
                id2entities[pid] = true_entities
    return id2entities


def add_entities(id2entities, input_file="./ts_note_all.pkl", output_file="./ts_note_rag.pkl"):
    data = pd.read_pickle(input_file)
    total_cnt = len(data)
    empty_cnt = 0
    for d in data:
        pid = d['PatientID']
        entities = id2entities.get(pid, [])
        if entities == []:
            empty_cnt += 1
        d['Entities'] = entities
    pd.to_pickle(data, output_file)
    print(f"缺失率: {empty_cnt / total_cnt}")


def peek_data(input_file):
    data = pd.read_pickle(input_file)
    pdb.set_trace()


if __name__ == "__main__":
    # id2e = merge()
    # add_entities(id2e)

    # peek_data('../final_splits/data_train_rag.pkl')  # 10776
    # peek_data('../final_splits/data_val_rag.pkl')  # 1539
    # peek_data('../final_splits/data_test_rag.pkl')  # 3080
    # peek_data('../mimic3_all/ts_note_all.pkl')  # 15395

    # peek_data('../mimic4_all/ts_note_all.pkl')  # 19331