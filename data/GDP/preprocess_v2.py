import os
import sys
from scipy.sparse import csr_matrix
import pickle
import networkx as nx
import numpy as np
from copy import deepcopy

dir_name = './result_v2'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

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



fname_drug = './gdp/drug_list.txt'
fp_drug = open(fname_drug, 'r')
lines = fp_drug.readlines()
drug_label_map = dict() # { Drug name : ID }
for i in range(len(lines)):
    drug_label_map[lines[i].replace('\n', '')] = i
fp_drug.close()

fname_cell = './gdp/cell_list.txt'
fp_cell = open(fname_cell, 'r')
lines = fp_cell.readlines()
cells = []
for l in lines:
    cells.append(l.replace('\n', ''))
fp_cell.close()

PROTEIN_PROTEIN_FILE = './gdp/IRefindex_protein_protein.txt'
CELL_PROTEIN_FILE = './gdp/IRefindex_cell_protein.txt'
protein_protein_file = open(PROTEIN_PROTEIN_FILE, 'r')
pplines = protein_protein_file.readlines()
cell_protein_file = open(CELL_PROTEIN_FILE, 'r')
cplines = cell_protein_file.readlines()

for cv in range(1, 11):
    node_names, _= load_embedding_from_txt("./gdp/EmbeddingData/ORI_NS/total_embedding-{}.txt".format(cv))
    n_nodes = len(node_names)
    cp_edge = np.zeros((n_nodes, n_nodes))
    pp_edge = np.zeros((n_nodes, n_nodes))
    cd_edge = np.zeros((n_nodes, n_nodes))

    node_map = dict() # { Node name : idx in feature matrix }

    print("Parse P-P info..")
    for i in range(int(len(pplines) / 2)):
        l = pplines[i * 2].split('\t')
        idx_1 = node_names.index(l[0])
        idx_2 = node_names.index(l[1])
        w = float(l[2])

        if l[0] not in node_map.keys():
            node_map[l[0]] = idx_1
        if l[1] not in node_map.keys():
            node_map[l[1]] = idx_2

        pp_edge[idx_1][idx_2] = w
        pp_edge[idx_2][idx_1] = w

    print("Parse C-P info..")
    for i in range(int(len(cplines) / 2)):
        l = cplines[i * 2].split('\t')
        idx_1 = node_names.index(l[0])
        idx_2 = node_names.index(l[1])
        w = float(l[2])

        if l[0] not in node_map.keys():
            node_map[l[0]] = idx_1
        if l[1] not in node_map.keys():
            node_map[l[1]] = idx_2
        cp_edge[idx_1][idx_2] = w
        cp_edge[idx_2][idx_1] = w

    print("Parse C-D info & nodes for training dataset..")
    train_cells_name = set()
    fname_cd = './gdp/fold/Cell-Drug-{}.txt'.format(cv)
    fp_cd = open(fname_cd, 'r')
    lines = fp_cd.readlines()
    for i in range(int(len(lines) / 2)):
        l = lines[i * 2].split('\t')
        idx_1 = node_names.index(l[0])
        idx_2 = node_names.index(l[1])
        w = float(l[2])

        if l[0] not in node_map.keys():
            node_map[l[0]] = idx_1
        if l[1] not in node_map.keys():
            node_map[l[1]] = idx_2
        cd_edge[idx_1][idx_2] = w
        cd_edge[idx_2][idx_1] = w

        train_cells_name.add(l[0])
    fp_cd.close()

    print("Save edges.pkl..")
    pp_edge = csr_matrix(pp_edge)
    cp_edge = csr_matrix(cp_edge)
    cd_edge = csr_matrix(cd_edge)
    #pp_edge.setdiag(1)
    #cp_edge.setdiag(1)
    #cd_edge.setdiag(1)
    edges = []
    edges.append(pp_edge)
    edges.append(cp_edge)
    edges.append(cd_edge)
    dir_cv = os.path.join(dir_name, 'FOLD-{}'.format(cv))
    if not os.path.exists(dir_cv):
        os.mkdir(dir_cv)
    with open(os.path.join(dir_cv, 'edges.pkl'), 'wb') as f:
        pickle.dump(edges, f)

    print("Nodes for test dataset..")
    test_cells_name = set()
    for i in range(1, 11):
        if i == cv:
            continue
        fname_cd = './gdp/fold/Cell-Drug-{}.txt'.format(cv)
        fp_cd = open(fname_cd, 'r')
        lines = fp_cd.readlines()
        for i in range(int(len(lines) / 2)):
            l = lines[i * 2].split('\t')
            idx_1 = node_names.index(l[0])
            if idx_1 not in test_cells_name:
                test_cells_name.add(l[0])
        fp_cd.close()

    # Set Labels
    train_cells_name = list(train_cells_name)
    sep = int(len(train_cells_name) * 0.9)
    train_cells = train_cells_name[:sep]
    valid_cells = train_cells_name[sep:]
    test_cells = list(test_cells_name)
    train_dict = dict()
    valid_dict = dict()
    test_dict = dict()

    train_idx = deepcopy(train_cells)
    for i in range(len(train_cells)):
        train_idx[i] = [train_idx[i], node_map[train_idx[i]]]
    train_idx.sort(key=lambda x:x[1])
    for i in range(len(train_idx)):
        train_dict[train_idx[i][0]] = i

    valid_idx = deepcopy(valid_cells)
    for i in range(len(valid_cells)):
        valid_idx[i] = [valid_idx[i], node_map[valid_idx[i]]]
    valid_idx.sort(key=lambda x:x[1])
    for i in range(len(valid_idx)):
        valid_dict[valid_idx[i][0]] = i

    test_idx = deepcopy(test_cells)
    for i in range(len(test_cells)):
        test_idx[i] = [test_idx[i], node_map[test_idx[i]]]
    test_idx.sort(key=lambda x:x[1])
    for i in range(len(test_idx)):
        test_dict[test_idx[i][0]] = i

    train_labels = np.zeros((len(train_cells), 265), dtype=np.float32)
    valid_labels = np.zeros((len(valid_cells), 265), dtype=np.float32)
    test_labels = np.zeros((len(test_cells), 265), dtype=np.float32)

    fname_cd = './gdp/fold/Cell-Drug-{}.txt'.format(cv)
    fp_cd = open(fname_cd, 'r')
    lines = fp_cd.readlines()
    for i in range(int(len(lines) / 2)):
        l = lines[i * 2].split('\t')
        idx_1 = node_map[l[0]]
        idx_2 = node_map[l[1]]
        if l[0] in train_cells:
            idx = train_dict[l[0]]
            train_labels[idx][drug_label_map[l[1]]] = 1
        else:
            idx = valid_dict[l[0]]
            valid_labels[idx][drug_label_map[l[1]]] = 1
    fp_cd.close()
    for i in range(1, 11):
        if i == cv:
            continue
        fname_cd = './gdp/fold/Cell-Drug-{}.txt'.format(cv)
        fp_cd = open(fname_cd, 'r')
        lines = fp_cd.readlines()
        for i in range(int(len(lines) / 2)):
            l = lines[i * 2].split('\t')
            if l[0] in test_cells:
                idx = test_dict[l[0]]
                test_labels[idx][drug_label_map[l[1]]] = 1
        fp_cd.close()

    print("Save labels.pkl..")
    train_labels = train_labels.tolist()
    valid_labels = valid_labels.tolist()
    test_labels = test_labels.tolist()
    labels = []
    labels.append(train_labels)
    labels.append(valid_labels)
    labels.append(test_labels)
    with open(os.path.join(dir_cv, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    print("Save nodes.pkl..")
    nodes = []
    train_nodes = []
    valid_nodes = []
    test_nodes = []
    for n in train_idx:
        train_nodes.append(n[1])
    for n in valid_idx:
        valid_nodes.append(n[1])
    for n in test_idx:
        test_nodes.append(n[1])
    sorted(train_nodes)
    sorted(valid_nodes)
    sorted(test_nodes)
    print("train_nodes : ", train_nodes)
    print("valid_nodes : ", valid_nodes)
    print("test_nodes : ", test_nodes)
    nodes.append(train_nodes)
    nodes.append(valid_nodes)
    nodes.append(test_nodes)
    with open(os.path.join(dir_cv, 'nodes.pkl'), 'wb') as f:
        pickle.dump(nodes, f)
