from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from sklearn import preprocessing
import torch
import time, os, json
import traceback
from string import punctuation

from config import pos_tagger, pos_dict_path, device, distance


class FlairDataset(object):

    def __init__(self, process_dict, base_model="deberta-base", base_model2='bert-base-multilingual-cased',
                 text_len=64, tag_name='O', multi_embed=False):
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
    def mutual_embed(embed1, embed2, distance=False):
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
            # complete_text = ' '.join(complete_text).lower().split()
            sentence = self.pad + complete_text + self.pad*(self.text_len + 2 - len(complete_text))
            sentence = ' '.join(sentence)
            # print("SENTENCE >>", sentence, len(sentence.split()))
            if self.multi_embed:
                embed1, embed2 = self.get_embedding(sentence)
                ids = self.mutual_embed(embed1, embed2)
            else:
                ids = self.get_embedding(sentence)[:self.text_len]
                ids = torch.stack(ids).to(torch.float)
            if ids is None or len(ids) < self.text_len:
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
            ids = torch.cat((ids, target_pos), dim=1)
            # print("999 >>", ids.shape, target_pos.shape)
        except:
            print("get item >>> ", traceback.print_exc())
            return None
        # print(item, "Item       finsih in : ", time.time()-start)
        return {
            "sentence": sentence,
            "ids": ids,
            "target_tag": torch.tensor(target_tag, dtype=torch.float),
            "target_pos": target_pos,
        }


class FlairDetection(FlairDataset):

    def __init__(self, process_dict={}, base_model="deberta-base", base_model2='bert-base-multilingual-cased',
                 text_len=64, tag_name='O', multi_embed=False):
        super(FlairDetection, self).__init__(process_dict=process_dict, base_model=base_model,
                                             base_model2=base_model2,
                                             text_len=text_len, tag_name=tag_name, multi_embed=multi_embed)
        # FlairDetection.__init__(process_dict=process_dict, base_model=base_model, text_len=text_len, tag_name=tag_name)

    def sentance_embeddings(self, sentence, experiment_name):
        # sentence = sentence.lower()
        sentence_ = self.pad[0] + ' ' + sentence + ' ' + ' '.join(self.pad * (self.text_len + 2 - len(sentence.split())))
        bert_emb, roberta_emb = self.get_embedding(sentence_)

        target_pos = [0]
        words = []
        i = 1
        sentence = Sentence(sentence_)
        for token in sentence:
            if i < self.text_len-1:
                target_pos.append(self.get_word_pos(token.text)/self.pos_num)
                words.append(
                    {
                        "text": token.text,
                        "start_offset": token.start_pos,
                        "end_offset": token.end_pos
                    }
                )

            else:
                break
            i+=1
        target_pos += [0]*(self.text_len-len(target_pos))
        target_pos = torch.tensor(target_pos, dtype=torch.float).to(device).reshape((self.text_len, 1))
        ids = None
        if 'bert' in experiment_name:
            ids = bert_emb
            ids = torch.stack(ids).to(torch.float)
        elif 'deberta' in experiment_name:
            ids = roberta_emb
            ids = torch.stack(ids).to(torch.float)
        elif 'mdm' in experiment_name:
            ids = self.mutual_embed(bert_emb, roberta_emb, distance=False)
        elif 'vsm' in experiment_name:
            ids = self.mutual_embed(bert_emb, roberta_emb, distance=True)

        if 'posam' in experiment_name:
            ids = torch.cat((ids, target_pos), dim=1)
            words.append('PAD')


        # if self.multi_embed:
        #     exp_emb = self.mutual_embed(embed1, embed2, distance)
        # elif 'bert' in experiment_name:
        #     ids = self.get_embedding(sentence_)[:self.text_len]
        #     ids = torch.stack(ids).to(torch.float)
        # if posam:
        #     ids = torch.cat((ids, target_pos), dim=1)

        return {
            "sentence": sentence_,
            "ids": ids,
            "words": words,
            "target_pos": target_pos/target_pos.sum(),
        }

