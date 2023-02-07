import os, traceback

from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from sklearn import preprocessing
import time, os, json
import traceback
from string import punctuation

import torch
from tqdm import tqdm
import joblib, os

from config import *
from .utils import create_dir
from .api_utils import *


class ConellStudy(object):

    def __init__(self):
        self.dataset_dir = conell_files_dir

    def read_file(self):
        final_report = {}
        data_files = os.listdir(self.dataset_dir)
        for data_file in data_files:
            # if 'train' in data_file:
            # print('\n', data_file)
            report = {'sentences': 0}
            with open(os.path.join(self.dataset_dir, data_file), "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in tqdm(lines, desc=data_file):
                    if line == '\n':
                        report['sentences'] += 1
                    else:
                        try:
                            entity, entity_tag = line.split('\t')
                            entity_tag = entity_tag.strip()
                            if entity_tag not in report.keys():
                                report[entity_tag] = 1
                            else:
                                report[entity_tag] += 1
                        except:
                            print(line, '>>>', traceback.print_exc())

            final_report[os.path.splitext(data_file)[0]] = report
        return final_report

    def analyse(self):
        pass


class DatasetAnalysis(object):

    def __init__(self):
        pass

    def report(self, target):
        dataset = ConellStudy()
        result = dataset.read_file()
        reports = {
            'target':0,
            'non_target':0
        }
        for section in result.keys():
            for label in result[section].keys():
                if target in label:
                    reports['target'] += result[section][label]
                else:
                    reports['non_target'] += result[section][label]
        print(result, '\n\n', reports)
        write_api_logs(str(reports))
        return reports


def process_conell_data(load_file, target_class):
    split_c = '\t' if 'conll' or 'relocar' in load_file  else ' '
    outputs = {'raw_words':[], 'raw_targets':[], 'entities':[], 'entity_tags':[], 'entity_spans':[]}
    with open(load_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # print("LINE >>> ", lines)
        raw_words, raw_targets = [], []
        raw_word, raw_target = [], []
        for line in lines:
            if line != "\n":
                raw_word.append(line.split(split_c)[0])
                raw_target.append(line.split(split_c)[1][:-1])
            else:
                raw_words.append(raw_word)
                for t_ in range(len(raw_target)):
                    # if raw_target[t_] != 'B-LOC':
                    #
                    if target_class not in raw_target[t_]:
                    # if '-ORG' not in raw_target[t_]:
                    # if '-PER' not in raw_target[t_]:
                        raw_target[t_] = 'O'
                raw_targets.append(raw_target)
                raw_word, raw_target = [], []
        # print("raw_targets =", raw_targets)
    for words, targets in zip(raw_words, raw_targets):
        entities, entity_tags, entity_spans = [], [], []
        start, end, start_flag = 0, 0, False
        for idx, tag in enumerate(targets):
            if tag.startswith('B-'):
                end = idx
                if start_flag:
                    entities.append(words[start:end])
                    entity_tags.append(targets[start][2:].lower())
                    entity_spans.append([start, end])
                    start_flag = False
                start = idx
                start_flag = True
            elif tag.startswith('I-'):
                end = idx
            elif tag.startswith('O'):
                end = idx
                if start_flag:
                    entities.append(words[start:end])
                    entity_tags.append(targets[start][2:].lower())
                    entity_spans.append([start, end])
                    start_flag = False
        if start_flag:
            entities.append(words[start:end+1])
            entity_tags.append(targets[start][2:].lower())
            entity_spans.append([start, end+1])
            start_flag = False

        if len(entities) != 0:
            outputs['raw_words'].append(words)
            outputs['raw_targets'].append(targets)
            outputs['entities'].append(entities)
            outputs['entity_tags'].append(entity_tags)
            outputs['entity_spans'].append(entity_spans)
    return outputs


class FlairExperimentalDataset(object):

    def __init__(self, process_dict, base_model="deberta-base", base_model2='bert-base-multilingual-cased',
                 text_len=64, tag_name='O', multi_embed=True):
        try:
            self.embed = TransformerWordEmbeddings(base_model)
            self.embed2 = None
            self.process_dict = process_dict
            self.text_len = text_len
            self.pos_num = 52
            self.noun_val = 51
            self.tag_name = tag_name
            self.enc_tag = preprocessing.LabelEncoder()
            self.nouns = ['NN', 'NNP', 'NNPS', 'NNS']
            self.pad = ['PAD']
            self.tagger = SequenceTagger.load(pos_tagger)
            self.multi_embed = multi_embed
            if self.multi_embed:
                self.embed2 = TransformerWordEmbeddings(base_model2)
        except Exception as e:
            write_api_logs(f"FlairExperimentalDataset init Exception:{str(e)}")

    def get_embedding(self, d_word):
        # create a sentence
        start = time.time()
        embeddings = []
        embeddings2 = []
        try:
            sentence = Sentence(d_word)
            # embed words in sentence
            self.embed.embed(sentence)
            # now get out the embedded tokens as tensor of 768 shape.
            for token in sentence:
                # print(d_word, "finish in ", time.time()-start)
                embeddings.append(token.embedding)
            if self.multi_embed:
                sentence2 = Sentence(d_word)
                if sentence2 is None:
                    write_api_logs(f"getembedding Exception: deberta embedding not get. None object received")
                    pass
                self.embed2.embed(sentence2)
                for token2 in sentence2:
                    embeddings2.append(token2.embedding)
                return embeddings[:self.text_len], embeddings2[:self.text_len]
            else:
                return embeddings[:self.text_len]
        except:
            print("get embeddings >>>", traceback.print_exc())
            return None

    @staticmethod
    def mutual_embed(embed1, embed2, distance=True):
        if distance:
            fls = open('loss.txt', 'a+')
            fls.write(f'{time.time()} - using distance embeddings.\n')
            fls.close()
            embed1 = torch.stack(embed1).to(torch.float)
            embed2 = torch.stack(embed2).to(torch.float)
            output = ((embed1 ** 2) + (embed2 ** 2)) ** 0.5
            embed3 = torch.max(embed1, embed2)
            embed3[embed3 >= 0] = 1
            embed3[embed3 < 0] = -1
            return embed3 * output
        else:
            # fls = open('loss.txt', 'a+')
            # fls.write(f'{time.time()} - without distance embeddings.\n')
            # fls.close()
            embed1 = torch.stack(embed1).to(torch.float)
            embed2 = torch.stack(embed2).to(torch.float)
            return (embed1 + embed2)/2

    def get_pos_dict(self, pos_dict={'<unk>': 0}, save=False):
        if not os.path.exists(pos_dict_path):
            save=True
        if save:
            if not os.path.exists(os.path.dirname(pos_dict_path)):
                os.makedirs(os.path.dirname(pos_dict_path))
            with open(pos_dict_path, 'w+') as op:
                json.dump(pos_dict, op)
            op.close()
        else:
            with open(pos_dict_path) as op:
                data = json.load(op)
            op.close()
            return data
        return None

    def get_word_pos(self, word):
        val = 0.0
        if str(word) in punctuation:
            return val
        pos_dict = self.get_pos_dict()
        try:
            sentence = Sentence(word)
            self.tagger.predict(sentence)
        except:
            return val

        try:
            for entity in sentence.get_labels('pos'):
                pos = entity.value
                if pos in self.nouns:
                    return self.noun_val
                if pos not in pos_dict.keys():
                    pos_dict[pos] = len(pos_dict)
                    self.get_pos_dict(pos_dict=pos_dict, save=True)
                return pos_dict[pos]
        except:
            return val
        return val

    def __len__(self):
        return len(self.process_dict[list(self.process_dict.keys())[0]])

    def __getitem__(self, item):
        start = time.time()

        def mark_tag(tag):
            if tag == self.tag_name:
                return 0
            else:
                return 1

        target_tag = []
        try:
            text = self.process_dict['raw_words'][item]
            tags = self.process_dict['raw_targets'][item]
            complete_text = text[: self.text_len - 1]
            complete_text = ' '.join(complete_text).split()
            sentence = self.pad + complete_text + self.pad*(self.text_len + 2 - len(complete_text))
            sentence = ' '.join(sentence)
            # print("SENTENCE >>", sentence, len(sentence.split()))

            bert_embed, roberta_embed = self.get_embedding(sentence)

            method1_embed = self.mutual_embed(bert_embed, roberta_embed, distance=False)
            method2_embed = self.mutual_embed(bert_embed, roberta_embed)

            bert_embed, roberta_embed = torch.stack(bert_embed).to(torch.float), torch.stack(roberta_embed).to(
                torch.float)

            if bert_embed is None or roberta_embed is None or method1_embed is None or method2_embed is None or \
                    len(bert_embed) < self.text_len:
                return None
            else:
                target_pos = [0]

                for i in range(self.text_len - 1):

                    if i < len(text):
                        target_pos.append(self.get_word_pos(text[i])/self.pos_num)
                    else:
                        target_pos.append(0.0)

            target_tag = list(map(mark_tag, tags))
            self.enc_tag.fit_transform(target_tag)

            target_tag = target_tag[:self.text_len - 2]
            target_tag = [0] + target_tag + [0]
            padding_len = self.text_len - len(target_tag)
            target_tag = target_tag + ([0] * padding_len)
            target_pos = torch.tensor(target_pos, dtype=torch.float).to(device).reshape((self.text_len, 1))
            bert_embed_with_pos = torch.cat((bert_embed, target_pos), dim=1)
            roberta_embed_with_pos = torch.cat((roberta_embed, target_pos), dim=1)
            method1_embed_with_pos = torch.cat((method1_embed, target_pos), dim=1)
            method2_embed_with_pos = torch.cat((method2_embed, target_pos), dim=1)
            # print("999 >>", ids.shape, target_pos.shape)
        except:
            print("get item >>> ", traceback.print_exc())
            return None
        # print(item, "Item       finsih in : ", time.time()-start)
        return {
            "sentence": sentence,
            "bert": bert_embed,
            "deberta": roberta_embed,
            "mdm": method1_embed,
            "vsm": method2_embed,
            "bert_posam": bert_embed_with_pos,
            "deberta_posam": roberta_embed_with_pos,
            "mdm_posam": method1_embed_with_pos,
            "vsm_posam": method2_embed_with_pos,
            "target_tag": torch.tensor(target_tag, dtype=torch.float),
            "target_pos": target_pos,
        }


class SaveDataset(object):

    def __init__(self, multi_embed=False):
        self.file_name = ['train.pkl', 'valid.pkl', 'test.pkl']
        self.multi_embed = multi_embed
        self.gen_path = flair_dataloader_tensors_path

    @staticmethod
    def conell_data(target_class):
        train_processed_data = process_conell_data(conell_train_file, target_class)
        valid_processed_data = process_conell_data(conell_valid_file, target_class)
        test_processed_data = process_conell_data(conell_test_file, target_class)
        return train_processed_data, valid_processed_data, test_processed_data

    def flair_experiment_dataloader_save(self, process_dict, batch_size=8, base_model=bert_base_model, base_model2=base_model,
                              text_len=64, tag_name='O', file_name="train.pkl", pickle_dirs=[]):
        # pickle_dirs = ['bert_embed', "roberta_embed", "mdm_embed", "vsm_embed",
        #                'bert_embed_with_pos', "roberta_embed_with_pos", "mdm_embed_with_pos", "vsm_embed_with_pos",]
        gen_path = self.gen_path

        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        try:
            flair_ds = FlairExperimentalDataset(process_dict=process_dict,
                                                base_model=base_model,
                                                base_model2=base_model2,
                                                text_len=64, tag_name='O',
                                                multi_embed=True)
        except Exception as e:
            write_api_logs(f"FlairExperimentalDataset Exception:{str(e)}")
            pass
        # result = {}
        for i in tqdm(range(0, len(flair_ds), batch_size), desc="DATASET >>>>  "):
            if i > 10:
                break
            sentences, target_tag, target_pos = [], [], []
            bert_embed, roberta_embed, method1_embed, method2_embed = [], [], [], []
            bert_embed_with_pos, roberta_embed_with_pos, method1_embed_with_pos, method2_embed_with_pos = [], [], [], []

            for j in tqdm(range(i, i + batch_size), desc="BATCH ***  "):
                j_batch = flair_ds[j]
                if j_batch is not None:
                    sentences.append(j_batch['sentence'])
                    target_tag.append(j_batch['target_tag'])
                    target_pos.append(j_batch['target_pos'])

                    bert_embed.append(j_batch['bert'])
                    roberta_embed.append(j_batch["deberta"])
                    method1_embed.append(j_batch["mdm"])
                    method2_embed.append(j_batch["vsm"])

                    bert_embed_with_pos.append(j_batch['bert_posam'])
                    roberta_embed_with_pos.append(j_batch["deberta_posam"])
                    method1_embed_with_pos.append(j_batch["mdm_posam"])
                    method2_embed_with_pos.append(j_batch["vsm_posam"])
                    # print(j_batch['ids'].shape, j_batch['target_tag'].shape, j_batch['target_pos'].shape)

            flair_batch = {
                "sentences": sentences,
                "target_tag": torch.stack(target_tag).to(torch.float),
                "target_pos": torch.stack(target_pos).to(torch.float),
                "bert": torch.stack(bert_embed).to(torch.float),
                "deberta": torch.stack(roberta_embed).to(torch.float),
                "mdm": torch.stack(method1_embed).to(torch.float),
                "vsm": torch.stack(method2_embed).to(torch.float),
                "bert_posam": torch.stack(bert_embed_with_pos).to(torch.float),
                "deberta_posam": torch.stack(roberta_embed_with_pos).to(torch.float),
                "mdm_posam": torch.stack(method1_embed_with_pos).to(torch.float),
                "vsm_posam": torch.stack(method2_embed_with_pos).to(torch.float),
            }
            for pickle_dir in pickle_dirs:
                flair_batch_dict = {
                    "sentences": sentences,
                    "target_tag": torch.stack(target_tag).to(torch.float),
                    "target_pos": torch.stack(target_pos).to(torch.float),
                    "ids": flair_batch[pickle_dir],
                }

                file_name_initial, file_name_ext = os.path.splitext(file_name)
                gen_pickle_path = os.path.join(os.path.join(gen_path, pickle_dir), file_name_initial)
                create_dir(gen_pickle_path)
                joblib.dump(flair_batch_dict,
                            os.path.join(gen_pickle_path, f"{i}.{file_name_ext}"),
                            compress=3)

    def unit_save(self, dataloaders, experiment_name, experiment=False):
        for i in range(len(dataloaders)):
            print(f"Saving file in Api:   {self.file_name[i]}")
            if experiment:
                self.flair_experiment_dataloader_save(
                    process_dict=dataloaders[i],
                    batch_size=batch_size,
                    base_model=bert_base_model,
                    base_model2=base_model,
                    text_len=text_len,
                    tag_name=tag_name,
                    file_name=self.file_name[i],
                    pickle_dirs=[experiment_name]
                )
            else:
                pass

    @staticmethod
    def add_dict(dict1, dict2):
        for k in dict1.keys():
            dict1[k].extend(dict2[k])
        return dict1

    @staticmethod
    def wikiann_data():
        train_processed_data = process_conell_data(wikiann_train_file)
        valid_processed_data = process_conell_data(wikiann_valid_file)
        test_processed_data = process_conell_data(wikiann_test_file)
        return train_processed_data, valid_processed_data, test_processed_data

    def save(self, dataset='conell', experiment_name='bert', target_class='-LOC', experiment=True):
        if dataset == 'conell':
            write_api_logs(f"Start Dataset processing \nExperiment:{experiment_name}\nTarget:{target_class}")
            train_processed_data, valid_processed_data, test_processed_data = self.conell_data(target_class=target_class)
            dataloaders = [train_processed_data, valid_processed_data, test_processed_data]
            self.unit_save(dataloaders, experiment_name=experiment_name, experiment=experiment)
        else:
            print('No such dataset exists.')
            write_api_logs('No such dataset exists.')
            return None


# if __name__ == '__main__':
#     da = DatasetAnalysis()
#     da.relocar_analysis()