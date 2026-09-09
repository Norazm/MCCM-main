from __future__ import print_function
import os
import os.path as osp
import time
import argparse
import logging
import hashlib
import copy
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from dataset import GraphDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import sparselearning

from sparselearning.core import Masking, CosineDecay
from models.model import GCNNet, GINNet, GATNet, SGCNet, APPNPNet, GCNIINet
# from models.model import GCNNet, SGCNet, APPNPNet, GCNIINet, GATNet, MLP, FAGCN, HGCN, LINK, GPRGNN, MixHop, FAGCNNet, HGCNNet
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

cudnn.benchmark = True
cudnn.deterministic = True


if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
if not os.path.exists('./results'): os.mkdir('./results')
logger = None

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

models = {}
models['gcn'] = (GCNNet)
models['gin'] = (GINNet)
models['gat'] = (GATNet)
# models['gat'] = (GATNet)
models['sgc'] = (SGCNet)
models['appnp'] = (APPNPNet)
models['gcnii'] = (GCNIINet)
# models['mlp'] = (MLP)
# models['fagcn'] = (FAGCN)
# models['h2gcn'] = (HGCN)
# models['link'] = (LINK)
# models['gprgnn'] = (GPRGNN)
# models['mixhop'] = (MixHop)
# models['fagcnnet'] = (FAGCNNet)
# models['h2gcnnet'] = (HGCNNet)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    if args.weight_sparse and not args.adj_sparse and not args.feature_sparse  :
        sparse_way = 'w'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}.log'.format(
            sparse_way,args.model,args.data, args.final_density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.adj_sparse and not args.weight_sparse and not args.feature_sparse :
        sparse_way = 'a'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}.log'.format(
            sparse_way,args.model,args.data, args.final_density_adj, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.feature_sparse and not args.weight_sparse and not args.adj_sparse :
        sparse_way = 'f'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}.log'.format(
            sparse_way,args.model,args.data, args.final_density_feature, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.weight_sparse and args.adj_sparse and not args.feature_sparse :
        sparse_way = 'wa'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}_{5}.log'.format(
            sparse_way,args.model,args.data, args.final_density, args.final_density_adj, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.weight_sparse and args.feature_sparse and not args.adj_sparse :
        sparse_way = 'wf'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}_{5}.log'.format(
            sparse_way,args.model,args.data, args.final_density, args.final_density_feature, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.adj_sparse and args.feature_sparse and not args.weight_sparse :
        sparse_way = 'af'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}_{5}.log'.format(
            sparse_way,args.model,args.data, args.final_density_adj, args.final_density_feature, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    elif args.weight_sparse and  args.adj_sparse and args.feature_sparse:
        sparse_way = 'waf'
        log_path = './logs/{0}/{1}_{2}_{3}_{4}_{5}_{6}.log'.format(
            sparse_way,args.model,args.data, args.final_density, args.final_density_adj, args.final_density_feature, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    else:
        sparse_way = 'base'
        log_path = './logs/{0}/{1}_{2}_{3}.log'.format(
            sparse_way,args.model,args.data, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    if not os.path.exists('./logs/{}'.format(sparse_way)): os.mkdir('./logs/{}'.format(sparse_way))

    #log_path = './logs/{0}/{1}_{2}_{3}_{4}_{5}.log'.format(sparse_way, args.model, args.data, args.final_density, args.final_density_adj, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def results_to_file(args, test_acc, train_time, test_time):
    if args.weight_sparse and not args.adj_sparse and not args.feature_sparse  :
        sparse_way = 'w'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.init_density, args.final_density)
    elif args.adj_sparse and not args.weight_sparse and not args.feature_sparse :
        sparse_way = 'a'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density_adj)
    elif args.feature_sparse and not args.weight_sparse and not args.adj_sparse :
        sparse_way = 'f'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density_feature)
    elif args.weight_sparse and args.adj_sparse and not args.feature_sparse :
        sparse_way = 'wa'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density, args.final_density_adj)
    elif args.weight_sparse and args.feature_sparse and not args.adj_sparse :
        sparse_way = 'wf'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density, args.final_density_feature)
    elif args.adj_sparse and args.feature_sparse and not args.weight_sparse :
        sparse_way = 'af'
        filename = "./results/{}/{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density_adj, args.final_density_feature)
    elif args.weight_sparse and  args.adj_sparse and args.feature_sparse:
        sparse_way = 'waf'
        filename = "./results/{}/{}_{}_{}_{}_{}_result.csv".format(
                            sparse_way, args.model, args.data, args.final_density, args.final_density_adj, args.final_density_feature)
    else:
        sparse_way = 'base'
        filename = "./results/{}/{}_{}_result.csv".format(
                            sparse_way, args.model, args.data)

    if not os.path.exists('./results/{}'.format(sparse_way)): os.mkdir('./results/{}'.format(sparse_way))

    headerList = ["Method","Growth","Prune Rate", "Update Frequency", "Final Prune Epoch", "::", "test_acc", "train_time", "test_time"]

    #filename = "./results/{}/{}_{}_{}_{}_result.csv".format(sparse_way, args.model, args.data, args.final_density, args.final_density_adj)
    with open(filename, "a+") as f:

        # reader = csv.reader(f)
        # row1 = next(reader)
        f.seek(0)
        header = f.read(6)
        if  header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                        fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, {}, {}, {}, :::,   {:.4f}, {:.4f}, {:.4f},{:.4f}, {:.4f}\n".format(
            args.method, args.growth_schedule, args.prune_rate, args.update_frequency, args.final_prune_epoch, test_acc, train_time, test_time
        )
        f.write(line)


def train(model, device, data, optimizer):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    data = data.to(device)
    target = data.y.reshape(-1).to(device)

    optimizer.zero_grad()

    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    optimizer.step()


def evaluate(model, device, data):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        logits = model(data)
        pred = logits.max(1)[1]
        gt = data.y.reshape(-1)

    return pred.cpu(), gt.cpu(), logits.cpu()


def calculate_metrics(preds, gts, logits, num_classes=3):
    # Confusion Matrix
    cm = confusion_matrix(gts, preds, labels=np.arange(num_classes))
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.sum(cm) - (fp + fn + tp)

    # ACC, SEN, SPE, PRE, F1
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1_scores = 2 * (precision * sensitivity) / (precision + sensitivity)

    average_accuracy = np.sum(tp) / np.sum(cm)
    average_sensitivity = np.mean(sensitivity)
    average_specificity = np.mean(specificity)
    average_precision = np.mean(precision)
    average_f1_score = np.mean(f1_scores)

    # AUROC
    auroc_scores = []
    gts = np.array(gts)
    logits = np.concatenate(logits, axis=0)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    for i in range(num_classes):
        binary_y = (gts == i).astype(int)
        auroc = roc_auc_score(binary_y, probabilities[:, i])
        auroc_scores.append(auroc)

    average_auroc_scores = np.mean(auroc_scores)

    # AUPRC
    auprc_scores = []
    for i in range(num_classes):
        binary_y = (gts == i).astype(int)
        auprc = average_precision_score(binary_y, probabilities[:, i])
        auprc_scores.append(auprc)

    average_auprc_scores = np.mean(auprc_scores)

    return cm, average_accuracy, average_sensitivity, average_specificity, average_precision, average_f1_score, average_auroc_scores, average_auprc_scores


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GraNet for sparse training')
    parser.add_argument('--data', type=str, default='CPTAC', choices=['CPTAC', 'TCGA', 'CPTAC2TCGA'])
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 100)')
    # parser.add_argument('--batch-size-jac', type=int, default=32, metavar='N', help='batch size for jac (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N', help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.00001)
    parser.add_argument('--lr_scheduler', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA training')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--save', type=str, default=''.join(str(time.time()).split('.')) + '.pt', help='path to save the final model')
    # parser.add_argument('--decay_frequency', type=int, default=25000)
    # parser.add_argument('--l1', type=float, default=0.0)
    # parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='gcnii', choices=['gcn', 'gat', 'gin', 'sgc', 'appnp', 'gcnii'])
    parser.add_argument('--dim', type=int, default=512, help='Feature dimensions of GNN layers.')
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    # parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    # parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    # parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')

    parser.add_argument('--adj_sparse', action='store_true', default=True, help='If Sparse Adj.')
    parser.add_argument('--feature_sparse', action='store_true', default=True, help='If Sparse Weight.')
    parser.add_argument('--weight_sparse', action='store_true', default=True, help='If Sparse Feature.')
    parser.add_argument('--decay-schedule', type=str, default='cosine', choices=['cosine', 'linear'], help='The decay schedule for the pruning rate. Default: cosine.')
    parser.add_argument('--growth_schedule', type=str, default='gradient', choices=['gradient', 'momentum', 'random'], help='The growth schedule. Default: gradient. Choose from: gradient, momentum, random.')
    
    # # FAGCN
    # parser.add_argument('--fagcn_layer_num', type=int, default=1)
    # parser.add_argument('--fagcn_dropout', type=float, default=0)
    # parser.add_argument('--fagcn_eps', type=float, default=0.1)

    # # MixHop
    # parser.add_argument('--mixhop_layer_num', type=int, default=1)
    # parser.add_argument('--mixhop_dropout', type=float, default=0)
    # parser.add_argument('--mixhop_hop', type=int, default=2)

    # # GPRGNN
    # parser.add_argument('--gprgnn_alpha', type=float, default=0.1)
    # parser.add_argument('--gprgnn_k', type=int, default=10)

    # # H2GCN
    # parser.add_argument('--h2gcn_dropout', type=float, default=0.1)

    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:{}'.format(args.cuda) if use_cuda else "cpu")

    print_and_log('\n')
    print_and_log('='*80)

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)

    for i in range(args.iters):
        #######################################################################################
        ############################# Datasets ################################################
        #######################################################################################
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        if args.data in ['CPTAC', 'TCGA', 'CPTAC2TCGA']:
            data_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.data)

            train_ids = open(osp.join(data_path, 'train.txt')).readlines()
            trainset = GraphDataset(data_path, train_ids, train_val='train')
            trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)

            test_ids = open(osp.join(data_path, 'test.txt')).readlines()
            testset = GraphDataset(data_path, test_ids, train_val='test')
            testloader = DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=True)

        #######################################################################################
        ############################# Models ################################################
        #######################################################################################

        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:
            if args.model == 'gcn':
                model = GCNNet(trainset, args).to(args.device)
            elif args.model == 'gat':
                model = GATNet(trainset, args).to(args.device)
            elif args.model == 'gin':
                model = GINNet(trainset, args).to(args.device)
            elif args.model == 'sgc':
                model = SGCNet(trainset, args).to(args.device)
            elif args.model == 'appnp':
                model = APPNPNet(trainset, args).to(args.device)
            elif args.model == 'gcnii':
                model = GCNIINet(trainset, args).to(args.device)


            print_and_log(model)
            print_and_log('='*60)
            print_and_log(args.model)
            print_and_log('='*60)
            print_and_log('Prune mode: {0}'.format(args.prune))
            print_and_log('Growth mode: {0}'.format(args.growth))
            print_and_log('Redistribution mode: {0}'.format(args.redistribution))
            print_and_log('='*60)


        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        if args.lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier], last_epoch=-1)
        else:
            lr_scheduler = None

        mask = None
        if args.sparse:
            decay = CosineDecay(args.prune_rate, (args.epochs*args.multiplier))
            mask = Masking(optimizer, prune_rate=args.prune_rate, death_mode=args.prune, prune_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args, train_loader=trainloader, device=args.device)
            mask.add_module(model, sparse_init=args.sparse_init)


        for epoch in range(1, args.epochs*args.multiplier + 1):

            #print_and_log("Epoch:{}".format(epoch))
            #print("="*50)

            # save models
            save_path = './save/' + str(args.model) + '/' + str(args.data) + '/' + str(args.method)
            save_subfolder = os.path.join(save_path, 'Multiplier=' + str(args.multiplier)  + '_sparsity' + str(1-args.final_density))
            if not os.path.exists(save_subfolder): os.makedirs(save_subfolder)

            t0 = time.time()
            for i_batch, sample_batched in enumerate(trainloader):
                train(model, args.device, sample_batched, optimizer)
            t1 = time.time()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if mask is not None:
                mask.step()

            all_preds, all_gts, all_logits  = [], [], []
            t2 = time.time()
            for i_batch, sample_batched in enumerate(testloader):
                preds, gts, logits = evaluate(model, args.device, sample_batched)
                all_preds.extend(preds.numpy())
                all_gts.extend(gts.numpy())
                all_logits.append(logits.numpy())
            t3 = time.time()

            cm, acc, sen, spe, pre, f1, auroc, auprc = calculate_metrics(all_preds, all_gts, all_logits)

            print_and_log('Epoch {} with {:.3f} seconds for training and {:.3f} seconds for testing.\
                          \n{}\
                          \naverage accuracy: \t\t{:.5f}\
                          \naverage sensitivity(recall): \t{:.5f}\
                          \naverage specificity: \t\t{:.5f}\
                          \naverage precision: \t\t{:.5f}\
                          \naverage f1_scores: \t\t{:.5f}\
                          \naverage auroc_score: \t\t{:.5f}\
                          \naverage auprc_score: \t\t{:.5f}'.format(epoch, t1-t0, t3-t2, cm, acc, sen, spe, pre, f1, auroc, auprc))

        # train_time_total = time.time() - t_start
        # print('Testing model')
        # model.load_state_dict(torch.load(os.path.join(save_subfolder, 'model_final.pth'))['state_dict'])

        # t_test_0 = time.time()

        # all_preds, all_gts = [], []
        # for i_batch, sample_batched in enumerate(testloader):
        #     preds, gts = evaluate(model, args.device, sample_batched)
        #     all_preds.extend(preds.numpy())
        #     all_gts.extend(gts.numpy())

        # cm = confusion_matrix(all_gts, all_preds, labels=np.arange(3))

        # results_to_file(args``, test_acc, train_time_total, time.time()- t_test_0)


if __name__ == '__main__':
    print("Start Runing!")
    main()
