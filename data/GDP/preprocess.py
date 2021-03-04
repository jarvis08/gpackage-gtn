import os
import sys
from scipy.sparse import csr_matrix
import pickle
import networkx as nx
import numpy as np


def load_embedding_from_txt(file_name):
    names = []
    embeddings = []
    with open(file_name, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            splitted = line.split()
            names.append(splitted[0])
            embeddings.append([float(value) for value in splitted[1:]])
    print(len(names)," words loaded.")
    return names, embeddings


DRUG_FILE = './gdp/drug_list.txt'
f_drug = open(DRUG_FILE, 'r')
drugs = f_drug.readlines()
drug_map = dict() # { Drug name : id }
for i in range(len(drugs)):
    drugs[i] = drugs[i].replace('\n', '')
    drug_map[drugs[i]] = i

cd_edges = dict() # { Cell name : [0, 1, 1, 0, ...] }
for cv in range(1, 11):
    CELL_DRUG_FILE = f'./gdp/fold/Cell-Drug-{cv}.txt'
    with open(CELL_DRUG_FILE, 'r') as f:
        lines = f.readlines()
        for i in range(int(len(lines) / 2)):
            l = lines[i * 2].split()
            c = l[0]
            d = l[1]
            if c not in cd_edges.keys():
                cd_edges[c] = [0 for _ in range(len(drugs))]
            cd_edges[c][drug_map[d]] = 1

cells = set(cd_edges.keys())
for cv in range(1, 11):
    node_names, _ = load_embedding_from_txt(f"./gdp/EmbeddingData/ORI_NS/total_embedding-{cv}.txt")
    id2name = dict()
    name2id = dict()
    for i in range(len(node_names)):
        id2name[i] = node_names[i]
        name2id[node_names[i]] = i

    CELL_DRUG_FILE = f'./gdp/fold/Cell-Drug-{cv}.txt'
    train_cells = set()
    with open(CELL_DRUG_FILE, 'r') as f:
        lines = f.readlines()
        for i in range(int(len(lines) / 2)):
            l = lines[i * 2].split()
            c = l[0]
            train_cells.add(c)
    test = list(cells - train_cells)
    train_cells = list(train_cells)
    split_point = int(len(train_cells) * 9 / 10)
    train = train_cells[:split_point]
    valid = train_cells[split_point:]

    #train_labels = [[0 for _ in range(len(drugs))] for _ in range(len(train))]
    #valid_labels = [[0 for _ in range(len(drugs))] for _ in range(len(valid))]
    #test_labels = [[0 for _ in range(len(drugs))] for _ in range(len(test))]
    #for i in range(len(train)):
    #    c = train[i]
    #    for d in cd_edges[c]:
    #        train_labels[i][drug_map[d]] = 1
    #for i in range(len(valid)):
    #    c = valid[i]
    #    for d in cd_edges[c]:
    #        valid_labels[i][drug_map[d]] = 1
    #for i in range(len(test)):
    #    c = test[i]
    #    for d in cd_edges[c]:
    #        test_labels[i][drug_map[d]] = 1

    #train_labels = [0 for _ in range(len(train))]
    #valid_labels = [0 for _ in range(len(valid))]
    #test_labels = [0 for _ in range(len(test))]
    #for i in range(len(train)):
    #    c = train[i]
    #    train_labels[i] = [name2id[c], cd_edges[c]]
    #for i in range(len(valid)):
    #    c = valid[i]
    #    valid_labels[i] = [name2id[c], cd_edges[c]]
    #for i in range(len(test)):
    #    c = test[i]
    #    test_labels[i] = [name2id[c], cd_edges[c]]

    train_nodes = [0 for _ in range(len(train))]
    valid_nodes = [0 for _ in range(len(valid))]
    test_nodes = [0 for _ in range(len(test))]
    train_labels = [0 for _ in range(len(train))]
    valid_labels = [0 for _ in range(len(valid))]
    test_labels = [0 for _ in range(len(test))]
    for i in range(len(train)):
        c = train[i]
        train_nodes[i] = name2id[c]
        train_labels[i] = cd_edges[c]
    for i in range(len(valid)):
        c = valid[i]
        valid_nodes[i] = name2id[c]
        valid_labels[i] = cd_edges[c]
    for i in range(len(test)):
        c = test[i]
        test_nodes[i] = name2id[c]
        test_labels[i] = cd_edges[c]
    nodes = []
    nodes.append(train_nodes)
    nodes.append(valid_nodes)
    nodes.append(test_nodes)
    labels = []
    labels.append(train_labels)
    labels.append(valid_labels)
    labels.append(test_labels)

    f_cd = open(CELL_DRUG_FILE, 'r')

    SAVE = f'./result/FOLD-{cv}'
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)
    with open(os.path.join(SAVE, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)
    with open(os.path.join(SAVE, 'nodes.pkl'), 'wb') as f:
        pickle.dump(nodes, f)


#node_names, node_features = load_embedding_from_txt(f"./gdp/EmbeddingData/ORI_NS/total_embedding-{cv}.txt")
#n_nodes = len(node_names)
#node_features = np.asarray(node_features)
#
#id2name = dict()
#name2id = dict()
#for i in range(len(node_names)):
#    id2name[i] = node_names[i]
#    name2id[node_names[i]] = i

#CELL_DRUG_FILE = f'./data/GDP/gdp/fold/Cell-Drug-{cv}.txt'
#cell_drug_file = open(CELL_DRUG_FILE, 'rb')
#cell_drug_graph = nx.read_edgelist(cell_drug_file, data=(('weight', int),))
#cell_drug_edges = nx.to_scipy_sparse_matrix(cell_drug_graph)
#G.add_edges_from(protein_protein_graph.edges, t='pp')
#G.add_edges_from(cell_protein_graph.edges, t='cp')
#G.add_edges_from(cell_drug_graph.edges, t='cd')
#cnt = 0
##for node in G.edges(data=True):
#for node in G.nodes:
#    if cnt > 3:
#        break
#    print(node)
#    cnt += 1
##tmp = nx.to_scipy_sparse_matrix(G, t='cc')
##print(tmp)

PROTEIN_PROTEIN_FILE = './gdp/IRefindex_protein_protein.txt'
CELL_PROTEIN_FILE = './gdp/IRefindex_cell_protein.txt'
protein_protein_file = open(PROTEIN_PROTEIN_FILE, 'r')
pplines = protein_protein_file.readlines()
cell_protein_file = open(CELL_PROTEIN_FILE, 'r')
cplines = cell_protein_file.readlines()
for cv in range(1, 11):
    node_names, node_features = load_embedding_from_txt(f"./gdp/EmbeddingData/ORI_NS/total_embedding-{cv}.txt")
    n_nodes = len(node_names)
    node_features = np.asarray(node_features)

    id2name = dict()
    name2id = dict()
    for i in range(len(node_names)):
        id2name[i] = node_names[i]
        name2id[node_names[i]] = i

    CELL_DRUG_FILE = f'./gdp/fold/Cell-Drug-{cv}.txt'
    cell_drug_file = open(CELL_DRUG_FILE, 'r')
    cdlines = cell_drug_file.readlines()
    cell_drug_file.close()
    #G = nx.MultiDiGraph()
    #for i in range(len(cplines)):
    #    l = cplines[i].split('\t')
    #    G.add_edge(name2id[l[0]], name2id[l[1]], weight=float(l[2]), etype="cp")
    #for i in range(len(pplines)):
    #    l = pplines[i].split('\t')
    #    G.add_edge(name2id[l[0]], name2id[l[1]], weight=float(l[2]), etype="pp")
    #for i in range(len(cdlines)):
    #    l = cdlines[i].split('\t')
    #    G.add_edge(name2id[l[0]], name2id[l[1]], weight=float(l[2]), etype="cd")
    cp_edge = np.zeros((n_nodes, n_nodes), dtype=float)
    pp_edge = np.zeros((n_nodes, n_nodes), dtype=float)
    cd_edge = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(len(cplines)):
        l = cplines[i].split('\t')
        cp_edge[name2id[l[0]]][name2id[l[1]]] = float(l[2])
        cp_edge[name2id[l[1]]][name2id[l[0]]] = float(l[2])
    for i in range(len(pplines)):
        l = pplines[i].split('\t')
        pp_edge[name2id[l[0]]][name2id[l[1]]] = float(l[2])
        pp_edge[name2id[l[1]]][name2id[l[0]]] = float(l[2])
    for i in range(len(cdlines)):
        l = cdlines[i].split('\t')
        cd_edge[name2id[l[0]]][name2id[l[1]]] = float(l[2])
        cd_edge[name2id[l[1]]][name2id[l[0]]] = float(l[2])
    pp = csr_matrix(pp_edge)
    cp = csr_matrix(cp_edge)
    cd = csr_matrix(cd_edge)
    pp.setdiag(1)
    cp.setdiag(1)
    cd.setdiag(1)

    edges = []
    edges.append(pp)
    edges.append(cp)
    edges.append(cd)

    SAVE = f'./result/FOLD-{cv}'
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)
    with open(os.path.join(SAVE, 'edges.pkl'), 'wb') as f:
        pickle.dump(edges, f)
