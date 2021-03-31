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
import logging
from time import time


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
    parser.add_argument('--epoch', type=int, default=4000,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    parser.add_argument('--cv', type=int, default=1,
                        help='Fold number of Cross Validation(10-fold)')
    parser.add_argument('--mode', type=str, default='test',
                        help='train or test')

    args = parser.parse_args()
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr
    cv = args.cv
    mode = args.mode

    logging.basicConfig(filename=f"Model/FOLD-{cv}.log", level=logging.DEBUG)
    logger = logging.getLogger()

    node_names, node_features = load_embedding_from_txt(f"./data/GDP/gdp/EmbeddingData/ORI_NS/total_embedding-{cv}.txt")
    node_features = np.asarray(node_features)

    with open(f'./data/GDP/result_v2/FOLD-{cv}/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open(f'./data/GDP/result_v2/FOLD-{cv}/labels.pkl','rb') as f:
        labels = pickle.load(f)
    with open(f'./data/GDP/result_v2/FOLD-{cv}/nodes.pkl','rb') as f:
        nodes = pickle.load(f)
    num_nodes = edges[0].shape[0]

    for i, edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    train_node = torch.from_numpy(np.array(nodes[0])).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])).type(torch.FloatTensor)

    valid_node = torch.from_numpy(np.array(nodes[1])).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])).type(torch.FloatTensor)

    test_node = torch.from_numpy(np.array(nodes[2])).type(torch.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])).type(torch.FloatTensor)

    num_classes = train_target.shape[1]
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
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        min_checkpoint = 2000
        patience = 200
        #min_checkpoint = 0
        #patience = 0
        fury = 0
        if mode == 'train':
            log = f">>> FOLD-{cv} Model Training"
            logger.debug(log)
            logger.debug(args)
            start_time = time()
            for i in range(epochs):
                for param_group in optimizer.param_groups:
                    if param_group['lr'] > 0.005:
                        param_group['lr'] = param_group['lr'] * 0.9
                print('Epoch:  ', i + 1)
                model.zero_grad()
                model.train()
                loss, y_train, Ws = model(A, node_features, train_node, train_target)
                train_f1 = micro_f1(torch.round(torch.sigmoid(y_train.detach())), train_target)
                print('Train - Loss: {}, F1-score: {:.4f}'.format(loss.detach().cpu().numpy(), train_f1))
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_loss, y_valid, _ = model.forward(A, node_features, valid_node, valid_target)
                    val_f1 = micro_f1(torch.round(torch.sigmoid(y_valid)), valid_target)
                    print('Valid - Loss: {}, F1-score: {:.4f}'.format(val_loss.detach().cpu().numpy(), val_f1))
                if i >= min_checkpoint:
                    if val_f1 > best_val_f1:
                        best_train_f1 = train_f1
                        best_val_f1 = val_f1
                        best_val_loss = val_loss.detach().cpu().numpy()
                        fury = 0

                        t = time() - start_time
                        if t > 3600:
                            t /= 3600
                            t = '{:.1f} hour'.format(t)
                        elif t > 60:
                            t /= 60
                            t = '{:.1f} min'.format(t)
                        else:
                            t = '{:.1f} sec'.format(t)
                        log = f'[Step: {i + 1}] Best Valid - F1-score: {best_val_f1:.4f}, Loss: {best_val_loss:.4f}, Time: ' + t
                        logger.debug(log)
                        print(log)

                        f_path = f'./Model/Model_FOLD-{cv}.pt'
                        logger.debug('Save model to ' + f_path)
                        torch.save(model, f_path)

                        #f_path = './Model/Model_state_dict_test.pt'
                        #logger.debug('Save to ' + f_path)
                        #torch.save(model.state_dict(), f_path)

                        #f_path = './Model/state_dict_and_optimizer.pt'
                        #logger.debug('Save to ' + f_path)
                        #torch.save({
                        #    'model': model.state_dict(),
                        #    'optimizer': optimizer.state_dict()
                        #    }, f_path)
                    else:
                        fury += 1
                    if fury >= patience:
                        break
                else:
                    if val_f1 > best_val_f1:
                        best_train_f1 = train_f1
                        best_val_f1 = val_f1
                        best_val_loss = val_loss.detach().cpu().numpy()

                        t = time() - start_time
                        if t > 3600:
                            t /= 3600
                            t = '{:.1f} hour'.format(t)
                        elif t > 60:
                            t /= 60
                            t = '{:.1f} min'.format(t)
                        else:
                            t = '{:.1f} sec'.format(t)
                        log = f'[Step: {i + 1}] Best Valid - F1-score: {best_val_f1:.4f}, Loss: {best_val_loss:.4f}, Time: ' + t
                        logger.debug(log)
                        print(log)
            print('---------------Best Results--------------------')
            print('Train - F1-score: {:.4f}'.format(best_train_f1))
            print('Valid - F1-score: {:.4f}'.format(best_val_f1))
            log = f"Final Validation F1-score: {best_val_f1:.4f}"
            logger.debug(log)
        else:
            print(f">>> FOLD-{cv} Model Test")
            #f_path = 'Model/Model_state_dict_test.pt'
            #model.load_state_dict(troch.load(f_path), strict=False)
            f_path = f'Model/Model_FOLD-{cv}.pt'
            model = torch.load(f_path)
            _, y_test, _ = model.forward(A, node_features, test_node, test_target)
            test_f1 = micro_f1(torch.round(torch.sigmoid(y_test)), test_target)
            print('Test - F1-score: {:.4f}'.format(test_f1))
