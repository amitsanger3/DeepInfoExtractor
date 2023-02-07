import torch
import traceback
from tqdm import tqdm
from config import *
from sklearn import model_selection
import os
import joblib
import random
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from .optimizer import BertAdam
from logs import Logger

# import numpy as np
# from sklearn.metrics import brier_score_loss

lmr_logger = Logger(job='lmr')


def loss_fn(output, target):
    # lfn = nn.CrossEntropyLoss()  # bce loss
    # lfn = torch.nn.BCELoss()
    # wt = torch.tensor([((target==0.).sum()/target.sum())]).to(device)
    # pos_wt1 = target
    # pos_wt1 = output*pos_t1

    # wt_target = target.clone()
    # wt_target[wt_target==1]=2
    # wt_target[wt_target==0]=1
    # wt_target[wt_target==2]=0

    # pos_wt2 = output*wt_target
    # pos_wt = target + wt_target*wt
    pos_wt = (target==0.0).sum()/target.sum()

    # print(f"\n\noutput:{output}\ntarget:{target}\npos_wt:{pos_wt}\n{'*'*100}\n\n")

    lfn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_wt)
    loss = lfn(output, target)
    return loss


def bias_params(output, target, beta=0.6):
    alpha = (target-output)/beta
    return alpha


def load_flair_dataloader(file_name):
    return joblib.load(os.path.join(flair_dataloader_tensors_path, file_name))


def merge_dataset(ds1, ds2):
    for k in ds1.keys():
        ds2[k].extend(ds1[k])
    return ds2


def train_fn3(data_loader_path, model, optim_schedule, device, batch_size, loss_f, epoch, lr=lr):
    # t_total = epoch * len(data_loader.keys()) * batch_size
    # optimizer = BertAdam(model, lr, t_total=t_total, b1=betas[0], b2=betas[1], max_grad_norm=clip)

    optimizer_defaults = dict(
        model=model, lr=lr, warmup=warmup, t_total=t_total, schedule=schedule,
        b1=betas[0], b2=betas[1], e=e, weight_decay=weight_decay,
        max_grad_norm=clip)
    optimizer = BertAdam(**optimizer_defaults)

    model.train()
    final_loss = 0
    model.zero_grad()
    data_files = os.listdir(data_loader_path)
    random.shuffle(data_files)
    for i, data_file in tqdm(enumerate(data_files), desc="TRAIN ITRS ***", total=len(data_files)):
        # res[int(data_file.split('.')[0])] = joblib.load(os.path.join(files_path, data_file))
    # kls = list(data_loader.keys())
    # random.shuffle(kls)
    # for i in tqdm(kls, desc="TRAIN ITRS ***", total=len(kls)):
        try:
            data = joblib.load(os.path.join(data_loader_path, data_file))
            x = data['ids'].to(device)
            y = data['target_tag'].to(device)
            z = data['target_pos'].to(device)
            if x.shape[0] < batch_size:
                continue
            pred = model(x, z)
            loss = loss_f(pred, y)
            x = x.detach()
            y = y.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            # loss = loss_f(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

            final_loss += loss.item()
            del loss
            del pred
            torch.cuda.empty_cache()
        except:
            print("TRAIN ERROR:\n", traceback.print_exc())
            continue
    return final_loss / len(data_files)


def calc_matrix(target, prediction, sentence, print_mat=False):

    tp, tn, fp, fn = 0,0,0,0
    thresh = 0.5

    # target[target == 1] = 2
    # target[target == 0] = 1
    # target[target == 2] = 0

    b,c = target.shape
    for i in range(b):
        pad = 0
        for j in range(c):
            targ = target[i,j]
            pred2 = prediction[i,j]
            if pad == 2:
                break
            # print(targ, pred2)
            if sentence[i].split()[j] == 'PAD':
                pad += 1
            if targ == 1 and pred2 >= thresh:
                tp +=1
            if targ == 1 and pred2 < thresh:
                fp +=1
            if targ == 0.0 and pred2 < thresh:
                tn +=1
            if targ == 0.0 and pred2 >= thresh:
                fn +=1

    N = sum([tp, tn, fp, fn])
    # N = b*c
    if print_mat:
        print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
    try:
        accuracy = (tp + tn)/N
    except:
        accuracy = 0.0
    try:
        pos_accuracy = tp/(tp+fp)
    except:
        pos_accuracy = 0.0
    try:
        precision = (1.0*tp)/(tp+fp)
    except:
        precision = 0.0
    try:
        recall = (1.0*tp)/(tp+fn)
    except:
        recall = 0.0
    try:
        f1 = 2.0/((1.0/precision)+(1.0/recall))
    except:
        f1 = 0.0
    return accuracy, pos_accuracy, precision, recall, f1


def eval_fn3(data_loader_path, model, device, batch_size, loss_f):
    model.eval()
    final_loss = 0
    accuracy_, pos_accuracy_, precision_, recall_, f1_ = 0,0,0,0,0
    data_files = os.listdir(data_loader_path)
    n = len(data_files)
    for i, data_file in tqdm(enumerate(data_files), desc="EVAL ITRS ***", total=n):
    # for i in tqdm(list(data_loader.keys()), desc="EVAL ITRS ***", total=len(list(data_loader.keys()))):
        with torch.no_grad():
            try:
                data = joblib.load(os.path.join(data_loader_path, data_file))
                x = data['ids'].to(device)
                y = data['target_tag'].to(device)
                z = data['target_pos'].to(device)
                st = data['sentences']
                if x.shape[0] < batch_size:
                    continue
                pred = model(x, z)
                # print(n, i, x.shape, y.shape, pred)
                loss = loss_f(pred, y)
                x = x.detach()
                z = z.detach()
                y = y.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()
                # loss = loss_f(pred, y)
                accuracy, pos_accuracy, precision, recall, f1 = calc_matrix(y, pred, st)
                accuracy_ += accuracy
                pos_accuracy_ += pos_accuracy
                precision_ += precision
                recall_ += recall
                f1_ += f1
                final_loss += loss.item()
                del loss
                del pred
                torch.cuda.empty_cache()
            except:
                print("VALIDATION ERROR:\n", traceback.print_exc())
                continue
    return final_loss/n, accuracy_/n, pos_accuracy_/n, precision_/n, recall_/n, f1_/n


def write_test_faulty_output(file_name, sentance, prod,faulty, n):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    fls = open(file_name, 'a+')
    fls.write(f"{n} {'-'*94}\n")
    fls.write(f"INPUT: {sentance}\n")
    fls.write(f" {faulty}\n")
    fls.write(f" {prod}\n")
    fls.write(f"  {'-' * 100}\n")
    fls.close()


def print_labels(y, pred, sentance, n, write_faulty_output=False):
    b, r = pred.shape
    for j in range(b):
        prod = []
        sentance_list = []
        faulty_list = []
        prod2 = []
        write_output = False
        for k in range(r):
            prod.append(f"{sentance[j].split()[k]} > {y[j][k]} > {pred[j][k]}")
            if write_faulty_output:
                if sentance[j].split()[k] != 'PAD':
                    sentance_list.append(sentance[j].split()[k])
                    prod2.append(f"{sentance[j].split()[k]} > {y[j][k]} > {pred[j][k]}")
                    if (float(y[j][k]) == 0.0 and float(pred[j][k]) >= 0.5) or \
                            (float(y[j][k]) == 1.0 and float(pred[j][k]) < 0.5):
                        faulty_list.append(f"{sentance[j].split()[k]} > {y[j][k]} > {pred[j][k]}")
                        write_output = True
        if write_output == True and write_faulty_output == True:
            write_test_faulty_output(file_name=faulty_op_path,
                                     sentance=' '.join(sentance_list),
                                     prod=' | '.join(prod2),
                                     faulty=' | '.join(faulty_list),
                                     n=n)
            n+=1
        print(" | ".join(prod))
    return n


def test_fn3(data_loader_path, model, device, batch_size, loss_f):
    model.eval()
    final_loss = 0
    accuracy_, pos_accuracy_, precision_, recall_, f1_ = 0,0,0,0,0
    data_files = os.listdir(data_loader_path)
    n = len(data_files)
    nl = 0
    for i, data_file in tqdm(enumerate(data_files), desc="TEST ITRS ***", total=n):
        with torch.no_grad():
            try:
                data = joblib.load(os.path.join(data_loader_path, data_file))
                x = data['ids'].to(device)
                y = data['target_tag'].to(device)
                z = data['target_pos'].to(device)
                # print('test tensor xxxx: ', x.shape, y.shape, z.shape)
                st = data['sentences']
                if x.shape[0] < batch_size:
                    continue
                pred = model(x, z)

                loss = loss_f(pred, y)
                x = x.detach()
                z = z.detach()
                y = y.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()
                # loss = loss_f(pred, y)
                nl = print_labels(y=y, pred=pred, sentance=st, write_faulty_output=True, n=nl)
                accuracy, pos_accuracy, precision, recall, f1 = calc_matrix(y, pred, st, print_mat=True)
                accuracy_ += accuracy
                pos_accuracy_ += pos_accuracy
                precision_ += precision
                recall_ += recall
                f1_ += f1
                final_loss += loss.item()
                del loss
                del pred
                torch.cuda.empty_cache()
            except:
                print("VALIDATION ERROR:\n", traceback.print_exc())
                continue
    return final_loss/n, accuracy_/n, pos_accuracy_/n, precision_/n, recall_/n, f1_/n


def detect_fn(data, model, device, batch_size):
    model.eval()
    with torch.no_grad():
        try:
            x = data['ids'].to(device)
            z = data['target_pos'].to(device)
            x = torch.stack([x for _ in range(batch_size)]).to(torch.float)
            z = torch.stack([z for _ in range(batch_size)]).to(torch.float)
            # print("0000000000", x.shape)
            pred = model(x, z)
            x = x.detach()
            z = z.detach()
            pred = pred.detach()
            torch.cuda.empty_cache()
        except:
            print("VALIDATION ERROR:\n", traceback.print_exc())
            return None
    return pred


def plot_figure(x, y, title, x_axis_title, y_axis_title, image_name):
    plt.figure(figsize=(20,10))
    for k in y.keys():
        plt.plot(x, y[k], label=k)
    plt.title(title, fontdict={'fontsize': 20})
    plt.xlabel(x_axis_title, fontsize = 14)
    plt.ylabel(y_axis_title, fontsize = 14)
    plt.legend(fontsize = 18)
    plt.grid()
    plt.savefig(image_name)
    plt.close()


def train_valid_processed_data(processed_data, distribution=0.25):
    if isinstance(processed_data, dict):
        train, valid = {}, {}
        for k in processed_data.keys():

            train[k], valid[k] = model_selection.train_test_split(processed_data[k],
                                                                  test_size=distribution, random_state=42)
    elif isinstance(processed_data, dict):
        train, valid = model_selection.train_test_split(processed_data, test_size=distribution, random_state=42)
    else:
        train, valid = {}, {}
    return train, valid


def custom_clustering_data(data_list):

    data = pd.concat(list(map(pd.DataFrame, data_list)))
    train, valid = model_selection.train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
    valid, test = model_selection.train_test_split(valid, test_size=0.5, random_state=42, shuffle=True)
    train, valid, test = train.to_dict('list'), valid.to_dict('list'), test.to_dict('list')
    return train, valid, test


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return None


def write_api_logs(log_text):
    fls = open(api_log_file, 'a+')
    msg = f"{datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')} || {log_text} \n"
    fls.write(msg)
    fls.close()