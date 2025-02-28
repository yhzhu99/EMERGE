import pandas as pd
import numpy as np
import random
import json
from collections import defaultdict, Counter
import pdb


def load_info(kg_file="/home/renchangyu-s22/Baichuan-7B-main/EHR_proj/with_rag/disease_features_clean.csv"):
    name2info = {}
    encode_key = 'mondo_definition'
    csv_df = pd.read_csv(kg_file)
    for idx, row in csv_df.iterrows():
        d = row.to_dict()
        rid = d['node_index']
        name = d['mondo_name']
        if not pd.isna(d[encode_key]): # 以encode_key为主，非空时再拼接其他字段
            info_str = f"[disease name]{d['mondo_name']} [definition]{d['mondo_definition']}. {d['orphanet_definition']} [description]{d['umls_description']}"
            name2info[name] = info_str
    return name2info


# def load_csv(csv_file):
#     with open(csv_file, 'r', encoding='utf8') as f:
#         data = pd.read_csv(f)
#     return data


def load_kg(disease_file="/home/renchangyu-s22/Baichuan-7B-main/EHR_proj/with_rag/disease_features_clean.csv",
            kg_file="/home/renchangyu-s22/Baichuan-7B-main/EHR_proj/with_rag/disease_subkg.csv"):
    # 疾病实体节点，包含详细属性信息
    disease_nodes = pd.read_csv(disease_file)  # 44133 nodes
    disease_nodes_plain = disease_nodes.to_json(orient='records')
    disease_nodes = json.loads(disease_nodes_plain)
    disease_name2id = {}  # 方便RAG，根据实体名称命中节点，再查询图谱中唯一标识符ID
    disease_id2detail = {}
    for node in disease_nodes:
        node_id = node['node_index']
        node_name = node['mondo_name']
        disease_name2id[node_name] = node_id
        node['as_head'] = []  # [disease_node] -> XXX
        node['as_tail'] = []  # XXX -> [disease_node]
        disease_id2detail[node_id] = node  # 后面的覆盖掉前面的，去重
    
    # 图谱中所有包含疾病实体节点（位于头/尾）的边
    disease_edges = pd.read_csv(kg_file)  # 618100 edges
    disease_edges_plain = disease_edges.to_json(orient='records')
    disease_edges = json.loads(disease_edges_plain)
    for idx, edge in enumerate(disease_edges):
        head_id = edge['x_index']
        head_name = edge['x_name']
        head_type = edge['x_type']
        if head_id in disease_id2detail.keys():
            disease_id2detail[head_id]['as_head'].append(edge)
        tail_id = edge['y_index']
        tail_name = edge['y_name']
        tail_type = edge['y_type']
        if tail_id in disease_id2detail.keys():
            disease_id2detail[tail_id]['as_tail'].append(edge)  # FIXME:看起来都是双向边，保留一个就行?
        relation = edge['display_relation']  # 相比relation更加细粒度
        if idx % 123456 == 0:
            print(f"EDGE {idx}: {head_name}({head_type}) -[{relation}]-> {tail_name}({tail_type})")
    return disease_name2id, disease_id2detail


def retrieval_by_name(name2info, name2id, id2detail, name):
    subkg_topk = 5
    def build_triple(edge:dict):
        return f"({edge['x_name']},{edge['display_relation']},{edge['y_name']})"
    
    node_id = name2id.get(name, -1)
    if node_id == -1:
        res = {}
    else:
        assert node_id in id2detail.keys()
        res = id2detail[node_id]
    node_desc = name2info.get(name, "")
    node_desc = node_desc.replace('nan', '')
    sub_graph_triples = [build_triple(edge) for edge in res['as_head']]
    sub_graph_triples = random.sample(sub_graph_triples, subkg_topk) if len(sub_graph_triples) > subkg_topk else sub_graph_triples[:subkg_topk]
    # pdb.set_trace()
    return node_desc, sub_graph_triples


def dump_data(name2info, name2id, id2detail, input_file, output_file):
    nodes_topk = 10  # 已经按出现频次降序排列了
    node_counter = Counter()
    data = pd.read_pickle(input_file)
    for item in data:
        nodes = item['Nodes']
        node_counter[len(nodes)] += 1
        nodes = random.sample(nodes, nodes_topk) if len(nodes) > nodes_topk * 2 else nodes[:nodes_topk]
        desc_list = []
        triples = []
        for node in nodes:
            desc, subkg = retrieval_by_name(name2info, name2id, id2detail, node)
            desc_list.append(desc)
            triples += subkg
        item['Documents'] = '\n'.join(desc_list)
        item['Triples'] = ','.join(triples)
        # pdb.set_trace()
    print(node_counter)
    pd.to_pickle(data, output_file)


if __name__ == "__main__":
    name2info = load_info()
    name2id, id2detail = load_kg()
    # test_name = "hyperreflexia (disease)"
    # retrieval_by_name(name2info, name2id, id2detail, test_name)
    dump_data(name2info, name2id, id2detail, './ts_note_node.pkl', './ts_note_kg.pkl')
