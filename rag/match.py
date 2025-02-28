import pandas as pd
from collections import defaultdict, Counter
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from get_bgem3_embedding import *
import pdb


def peek_data(input_file):
    data = pd.read_pickle(input_file)
    all_entities = Counter()
    for item in data:
        es = item['Entities']
        for e in es:
            all_entities[e] += 1
    # pdb.set_trace()  # 251779 -> 53934 unique entities
    return [_ for _ in all_entities.keys()]


def batch_match(entities, chunk_size=100, specified_thresh=None):
    nodes = []
    device = torch.device('cuda', device_no) if torch.cuda.is_available() else torch.device('cpu')
    # get kg feature in total
    kg_feature, kg_map = load_kg_feature('../with_rag/entityname_map.txt', '../with_rag/kg_entityname_bgem3.pkl') # entityname的feature用于做entity->node匹配，entityinfo_map用于匹配后返回对应百科
    iterations = len(entities) // chunk_size
    if len(entities) % chunk_size != 0:
        iterations += 1
    for i in tqdm([_ for _ in range(iterations)]):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        s = entities[start_index:end_index]
        sen_emb = get_feature(s)
        sen_emb = torch.Tensor(sen_emb).to(device)
        kg_infos, sen_retrieval_features, top1_sim = retrieve_related_kg_fast(sen_emb, kg_feature, kg_map, debug=False, specified_thresh=specified_thresh)
        nodes += kg_infos
    results = {k:v for k, v in zip(entities, nodes)}
    return results


def dump_data(entity2node, input_file, output_file):
    data = pd.read_pickle(input_file)
    for item in data:
        es = item['Entities']
        nodes = []
        for e in es:
            node = entity2node.get(e, '')
            if node != '':
                nodes.append(node)
        new_nodes = []
        for _ in nodes:  # 保持原有顺序
            if _ not in new_nodes:
                new_nodes.append(_)
        item['Nodes'] = new_nodes
        # pdb.set_trace()
    pd.to_pickle(data, output_file)


if __name__ == "__main__":
    all_entities = peek_data('./ts_note_rag.pkl')
    results = batch_match(all_entities)
    # print(results)
    dump_data(results, './ts_note_rag.pkl', './ts_note_node.pkl')
    