import torch
import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model_gdp import GTN
import pdb
import pickle
import argparse
from utils import f1_score


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
    print(len(names)," nodes loaded.")
    return names, embeddings


def micro_f1(logits, labels):
    #predicted = tf.cast(predicted, dtype=tf.int32)
    #labels = tf.cast(self.labels, dtype=tf.int32)

    #true_pos = tf.math.count_nonzero(predicted * labels)
    #false_pos = tf.math.count_nonzero(predicted * (labels - 1))
    #false_neg = tf.math.count_nonzero((predicted - 1) * labels)

    #precision = true_pos / (true_pos + false_pos)
    #recall = true_pos / (true_pos + false_neg)
    #fmeasure = (2 * precision * recall) / (precision + recall)

    predicted = logits.type(torch.IntTensor)
    labels = labels.type(torch.IntTensor)

    true_pos = torch.count_nonzero(predicted * labels)
    false_pos = torch.count_nonzero(predicted * (labels - 1))
    false_neg = torch.count_nonzero((predicted - 1) * labels)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    fmeasure = (2 * precision * recall) / (precision + recall)
    return fmeasure.type(torch.FloatTensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="GDP",
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    parser.add_argument('--cv', type=int, default=1,
                        help='Fold number of Cross Validation(10-fold)')

    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr
    cv = args.cv

    node_names, node_features = load_embedding_from_txt(f"./data/GDP/gdp/EmbeddingData/ORI_NS/total_embedding-{cv}.txt")
    node_features = np.asarray(node_features)

    with open('data/' + args.dataset + f'/result/FOLD-{cv}/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open('data/' + args.dataset + f'/result/FOLD-{cv}/labels.pkl','rb') as f:
        labels = pickle.load(f)
    with open('data/' + args.dataset + f'/result/FOLD-{cv}/nodes.pkl','rb') as f:
        nodes = pickle.load(f)
    num_nodes = edges[0].shape[0]
    for i,edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    train_node = torch.from_numpy(np.array(nodes[0])).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])).type(torch.FloatTensor)
    valid_node = torch.from_numpy(np.array(nodes[1])).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])).type(torch.FloatTensor)
    test_node = torch.from_numpy(np.array(nodes[2])).type(torch.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])).type(torch.FloatTensor)
    print(train_node.shape)
    print(train_target.shape)

    num_classes = train_target.shape[1]
    final_f1 = 0
    for l in range(1):
        model = GTN(num_edge=A.shape[-1],
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_class=num_classes,
                            num_layers=num_layers,
                            norm=norm)
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
        else:
            optimizer = torch.optim.Adam([{'params':model.weight},
                                        {'params':model.linear1.parameters()},
                                        {'params':model.linear2.parameters()},
                                        {"params":model.layers.parameters(), "lr":0.5}
                                        ], lr=0.005, weight_decay=0.001)
        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        
        for i in range(epochs):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ',i+1)
            model.zero_grad()
            model.train()
            loss, y_train, Ws = model(A, node_features, train_node, train_target)

            train_f1 = micro_f1(torch.round(y_train.detach()), train_target)
            #train_f1 = torch.mean(f1_score(torch.round(y_train.detach()), train_target, num_classes=2)).cpu().numpy()
            #train_f1 = torch.mean(f1_score(torch.round(y_train.detach()), train_target, num_classes=num_classes)).cpu().numpy()
            #train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1))
            loss.backward()
            optimizer.step()
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid, _ = model.forward(A, node_features, valid_node, valid_target)
                val_f1 = micro_f1(torch.round(y_valid), valid_target)
                #val_f1 = torch.mean(f1_score(torch.round(y_valid), valid_target, num_classes=2)).cpu().numpy()
                #val_f1 = torch.mean(f1_score(torch.round(y_valid), valid_target, num_classes=num_classes)).cpu().numpy()
                #val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                test_loss, y_test, W = model.forward(A, node_features, test_node, test_target)
                test_f1 = micro_f1(torch.round(y_test), test_target)
                #test_f1 = torch.mean(f1_score(torch.round(y_test), test_target, num_classes=2)).cpu().numpy()
                #test_f1 = torch.mean(f1_score(torch.round(y_test), test_target, num_classes=num_classes)).cpu().numpy()
                #test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                print('Test - Loss: {}, Macro_F1: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1))
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1 
        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
        final_f1 += best_test_f1
