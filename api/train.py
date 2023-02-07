# from display import create_report
import numpy as np
import torch
import joblib, os, sys
import json
from tqdm import tqdm
from torch.optim import SGD

from .utils import train_fn3, eval_fn3, test_fn3, create_dir, loss_fn, write_api_logs
from config import *
from .model import LMRBiLSTMAttnCRF
from shedular import ScheduledOptim


class ExperimentalLearning(object):

    def __init__(self):
        self.learning_pattern = {
            "initial_learning": {'epochs': [0,20], 'learning_rate': 1e-3},
            "deep_learning": {'epochs': [20,51], 'learning_rate': 1e-5}
        }
        self.motitoring_file = monitoring_file

    def model(self, embed_size, hidden_size, rnn_layers, device):
        model = LMRBiLSTMAttnCRF(
            embedding_size=embed_size,
            hidden_dim=hidden_size,
            rnn_layers=rnn_layers,
            lstm_dropout=0.3,
            device=device, key_dim=64, val_dim=64,
            num_output=64,
            num_heads=16,
            attn_dropout=0.3  # changed from 0.3
        )
        return model

    def trainable_params_info(self, model):
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total Params:", pytorch_total_params)
        write_api_logs(f"Total Params: {str(pytorch_total_params)}")

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Trainable Params:", pytorch_total_params)
        write_api_logs(f"Total Trainable Params: {str(pytorch_total_params)}")

    def load_flair_dataloader(self, files_path):
        res = {}
        data_files = os.listdir(files_path)
        for data_file in data_files:
            res[int(data_file.split('.')[0])] = joblib.load(os.path.join(files_path, data_file))
        return res

    def monitoring_save_file(self, data={}):
        with open(self.motitoring_file, 'w+') as op:
            json.dump(data, op)
        op.close()

    def monitoring_load_file(self):
        if not os.path.exists(self.motitoring_file):
            self.monitoring_save_file()
        with open(self.motitoring_file, 'r') as op:
            data = json.load(op)
        op.close()
        return data

    def reduce_learning(self):
        experimental_datasets = os.listdir(flair_dataloader_tensors_path)
        for experimental_dataset in experimental_datasets:
            print('Starting experiments on ..... ', experimental_dataset)
            write_api_logs(f'Starting experiments on ..... {str(experimental_dataset)}')
            print("Loading training & validation dataset...")
            write_api_logs("Loading training & validation dataset...")
            # train_data_loader = self.load_flair_dataloader(
            #     files_path=os.path.join(os.path.join(flair_dataloader_tensors_path, experimental_dataset), 'train')
            # )
            # valid_data_loader = self.load_flair_dataloader(
            #     files_path=os.path.join(os.path.join(flair_dataloader_tensors_path, experimental_dataset), 'valid')
            # )
            # test_data_loader = self.load_flair_dataloader(
            #     files_path=os.path.join(os.path.join(flair_dataloader_tensors_path, experimental_dataset), 'test')
            # )
            # print(len(train_data_loader.keys()), len(valid_data_loader.keys()), device, batch_size)

            experimental_model_path = os.path.join(model_path, experimental_dataset)
            # experimental_report_path = os.path.join(report_path, experimental_dataset)
            experimental_logs_path = os.path.join(logs_path, experimental_dataset)

            create_dir(experimental_model_path)
            # create_dir(experimental_report_path)
            create_dir(experimental_logs_path)

            if 'posam' in experimental_dataset:
                model = self.model(embed_size=embed_size, hidden_size=hidden_size,
                                   rnn_layers=rnn_layers, device=device)
            else:
                without_pos_embed = embed_size-1
                model = self.model(embed_size=without_pos_embed, hidden_size=hidden_size,
                                   rnn_layers=rnn_layers, device=device)
            print(f"{'*' * 25} {experimental_dataset} {'*' * 25}")
            self.trainable_params_info(model)
            print(f"{'*' * 100}")
            for pattern in self.learning_pattern.keys():

                # ####################### REPETITION CHECK #############
                monitoring_data = self.monitoring_load_file()
                if experimental_dataset in monitoring_data.keys():
                    if pattern in monitoring_data[experimental_dataset]:
                        continue
                else:
                    monitoring_data[experimental_dataset] = []

                # ######################## TITLE ########################
                title = f"BDRNN model experimenting on {experimental_dataset} with {pattern} training."

                start_epoch, num_epochs = self.learning_pattern[pattern]['epochs']
                lr = self.learning_pattern[pattern]['learning_rate']
                fls = open(os.path.join(experimental_logs_path, "RNN_FLAIR_N2.txt"), "a+")
                fls.write(f"{'_'*25} {pattern}  {'_'*25}")
                fls.close()
                if pattern == "initial_learning":
                    print("Current training: {} will have learning rate: {}".format(pattern, lr))
                    write_api_logs("Current training: {} will have learning rate: {}".format(pattern, lr))
                    model.to(device)
                else:
                    print("Current training: {} will have learning rate: {}".format(pattern, lr))
                    write_api_logs("Current training: {} will have learning rate: {}".format(pattern, lr))
                    print("Pretrained model in loading...")
                    write_api_logs("Pretrained model in loading...")
                    model_name = os.path.join(experimental_model_path, "lmr.pt")
                    if not os.path.exists(model_name):
                        print("""Pretrained model required to start the training at start_epoch > 0. System consider that you 
                                          want to train further a model that you previously trained at particular epochs. If you are on 
                                          fresh start then set start_epoch = 0 in the argument or leave for default. """)
                        write_api_logs("""Pretrained model required to start the training at start_epoch > 0. System consider that you 
                                                                  want to train further a model that you previously trained at particular epochs. If you are on 
                                                                  fresh start then set start_epoch = 0 in the argument or leave for default. """)
                        sys.exit()
                    model.load_state_dict(torch.load(model_name, map_location=device))
                    model.to(device)
                optim = SGD(model.parameters(), lr=lr, momentum=0.09)
                optim_schedule = ScheduledOptim(optim, 8, n_warmup_steps=warmup_steps)

                best_loss = np.inf
                train_loss_list = []
                valid_loss_list = []

                accuracy_list, pos_accuracy_list, precision_list, recall_list, f1_list = [], [], [], [], []

                # model.to(torch.device(device))

                for epoch in tqdm(range(start_epoch, num_epochs), desc="EPOCHS ---"):
                    model.zero_grad()
                    train_loss = train_fn3(data_loader_path=os.path.join(os.path.join(flair_dataloader_tensors_path, experimental_dataset), 'train'),
                                           model=model,
                                           optim_schedule=optim_schedule,
                                           device=device,
                                           batch_size=batch_size,
                                           loss_f=loss_fn,
                                           epoch=epoch, lr=lr)
                    test_loss, accuracy, pos_accuracy, precision, recall, f1 = eval_fn3(
                        data_loader_path=os.path.join(os.path.join(flair_dataloader_tensors_path, experimental_dataset), 'valid'),
                        model=model,
                        device=device,
                        batch_size=batch_size,
                        loss_f=loss_fn)
                    train_loss_list.append(train_loss)
                    valid_loss_list.append(test_loss)
                    accuracy_list.append(accuracy)
                    pos_accuracy_list.append(pos_accuracy)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)

                    # create_report(train_loss_list=train_loss_list,
                    #               valid_loss_list=valid_loss_list,
                    #               title=title, report_path=os.path.join(experimental_report_path, pattern),
                    #               accuracy_list=accuracy_list,
                    #               pos_accuracy_list=pos_accuracy_list,
                    #               precision_list=precision_list,
                    #               recall_list=recall_list,
                    #               f1_list=f1_list)

                    print(
                        f"\nEpoch = {epoch} Train Loss = {train_loss} Valid Loss = {test_loss} Accuracy = {accuracy}")
                    fls = open(os.path.join(experimental_logs_path, "RNN_FLAIR_N2.txt"), "a+")
                    fls.write(
                        f"Epoch = {epoch} | Train Loss = {train_loss} | Valid Loss = {test_loss} | Accuracy = {accuracy}\
                                             | POS Accuracy = {pos_accuracy} | \
                                        Precision = {precision} | Recall = {recall} | F1 = {f1} \n")
                    fls.close()
                    write_api_logs(f"Epoch = {epoch} | Train Loss = {train_loss} | Valid Loss = {test_loss} | Accuracy = {accuracy}\
                    Precision = {precision} | Recall = {recall} | F1 = {f1} \n")
                    if test_loss < best_loss:
                        torch.save(model.state_dict(), os.path.join(experimental_model_path, f"lmr.pt"))
                    best_loss = test_loss

            # start model testing

                model_name = os.path.join(experimental_model_path, "lmr.pt")

                model.load_state_dict(torch.load(model_name, map_location=device))
                model.to(device)
                print(model.eval())
                print("Loading testing dataset...")
                write_api_logs("Loading testing dataset...")
                test_loss, accuracy, pos_accuracy, precision, recall, f1 = test_fn3(data_loader_path=os.path.join(os.path.join(flair_dataloader_tensors_path, experimental_dataset), 'test'),
                                                                                    model=model,
                                                                                    device=device,
                                                                                    batch_size=batch_size,
                                                                                    loss_f=loss_fn)
                print(
                    f"Test loss: {test_loss}, Accuracy:{accuracy}, \nPOS Accuracy:{pos_accuracy}, \nPrecesion:{precision},\
                             Recall:{recall}, F1:{f1}")
                fls = open(os.path.join(experimental_logs_path, "RNN_FLAIR_N2.txt"), "a+")
                fls.write("\n----------------Testing model for {}-----------------\n".format(pattern))
                fls.write(
                    f"Test loss: {test_loss}, Accuracy:{accuracy}, \nPOS Accuracy:{pos_accuracy}, \nPrecesion:{precision},\
                             Recall:{recall}, F1:{f1}\n")
                fls.close()
                write_api_logs(f"Test loss: {test_loss}, Accuracy:{accuracy}, \nPrecesion:{precision},\
                             Recall:{recall}, F1:{f1}\n")
                monitoring_data[experimental_dataset].append(pattern)
                self.monitoring_save_file(data=monitoring_data)


