from config import *

from .utils import detect_fn
from .model import LMRBiLSTMAttnCRF
from .dataset import FlairDetection

import torch
import re


class MicroBlogAnalysis(object):

    def __init__(self):
        self.model = None
        self.model_name = None
        # self.model_name = os.path.join(api_model_path, "lmr.pt")
        # self.organization_model_name = os.path.join(api_organization_model_path, "lmr.pt")
        # self.person_model_name = os.path.join(api_person_model_path, "lmr.pt")
        # self.fashion_model_name = os.path.join(api_fashion_model_path, "lmr.pt")
        # self.chemical_model_name = os.path.join(api_chemical_model_path, "lmr.pt")
        # self.diese_model_name = os.path.join(api_disease_model_path, "lmr.pt")
        # self.model_name = os.path.join(api_model_path, "lmr.pt")

    def get_model(self, model_path, embed_size):
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
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model

    def locations(self, blog, blog_id, experiment_name, MULTI_EMBEDDING):
        results = {
            "message": None,
            "tweet_id": blog_id,
            "location_mentions": []
        }
        print("---", blog)
        if len(blog) < 2:
            print("""You must enter a valid micro blog sentence have a before detecting location. """)
            results['message'] = """You must enter a valid micro blog sentence have a before detecting location. """
            return results
        self.model_name = os.path.join(os.path.join(model_path, experiment_name), "lmr.pt")
        # if 'mdm' in experiment_name:
        #     distance = True
        # else:
        #     distance = False

        if not os.path.exists(self.model_name):
            print("""Pretrained model required for location detection. Please train your weights first. """)
            results['message'] = """Pretrained model required for location detection. 
                                    Please train your weights first. """
            return results
        if 'posam' in experiment_name:
            model = self.get_model(self.model_name, embed_size=embed_size)
        else:
            model = self.get_model(self.model_name, embed_size=embed_size-1)

        # if MULTI_EMBEDDING:
        fd = FlairDetection(base_model=bert_base_model, base_model2=base_model, multi_embed=True)
        # else:
        #     fd = FlairDetection(base_model=bert_base_model, base_model2=base_model, multi_embed=False)
        fdr = fd.sentance_embeddings(blog, experiment_name)
        predictions = detect_fn(data=fdr, model=model, device=device, batch_size=batch_size)

        for i in range(len(fdr["words"])):
            prediction_op = predictions[0][i].item()
            print("       ", fdr["words"][i], prediction_op)
            try:
                word_ = fdr["words"][i]
                word = word_["text"]
            except:
                continue
            if prediction_op > 0.5 and word != 'PAD' and len(word)>1:
                # res = re.search(word, blog)
                word_start = word_["start_offset"]
                word_end = word_["end_offset"]
                if len(results["location_mentions"]) > 0:
                    prev = results["location_mentions"].pop()
                    prev_word = prev["text"]
                    prev_start = prev["start_offset"]
                    prev_end = prev["end_offset"]

                    if prev_end == word_start:
                        word = prev_word + word
                        word_start = prev_start

                    elif prev_end + 1 == word_start:
                        word = prev_word + ' ' + word
                        word_start = prev_start

                    else:
                        results["location_mentions"].append(prev)

                results["location_mentions"].append({
                    "text": word,
                    "start_offset": word_start,
                    "end_offset": word_end
                })
        print(results)
        return results

    def get_results(self, blog, blog_id, model_path_name):
        results = {
            "tweet_id": blog_id,
            "location_mentions": []
        }
        print("---", blog)
        if len(blog) < 2:
            print("""You must enter a valid micro blog sentence have a before detecting location. """)
            return results
        if not os.path.exists(model_path_name):
            print("""Pretrained model required for location detection. Please train your weights first. """)
            return results
        model = self.get_model(model_path_name)

        if MULTI_EMBEDDING:
            fd = FlairDetection(base_model=bert_base_model, base_model2=base_model, multi_embed=True)
        else:
            fd = FlairDetection(base_model=bert_base_model)
        fdr = fd.sentance_embeddings(blog)
        predictions = detect_fn(data=fdr, model=model, device=device, batch_size=batch_size)

        for i in range(len(fdr["words"])):
            prediction_op = predictions[0][i].item()
            print("       ", fdr["words"][i], prediction_op)
            try:
                word_ = fdr["words"][i]
                word = word_["text"]
            except:
                continue
            if prediction_op > 0.5 and word != 'PAD' and len(word)>1:
                # res = re.search(word, blog)
                word_start = word_["start_offset"]
                word_end = word_["end_offset"]
                # preds = [str(prediction_op)]
                if len(results["location_mentions"]) > 0:
                    prev = results["location_mentions"].pop()
                    prev_word = prev["text"]
                    prev_start = prev["start_offset"]
                    prev_end = prev["end_offset"]

                    if prev_end == word_start:
                        word = prev_word + word
                        word_start = prev_start
                        # preds = [prev['probability']] + preds

                    elif prev_end + 1 == word_start:
                        word = prev_word + ' ' + word
                        word_start = prev_start
                        # preds = [prev['probability']] + preds

                    else:
                        results["location_mentions"].append(prev)

                results["location_mentions"].append({
                    "text": word,
                    "start_offset": word_start,
                    "end_offset": word_end,
                    # "probability": ' '.join(preds)
                })
        print(results)
        return results

    # def persons(self, blog, blog_id):
    #     return self.get_results(blog=blog,
    #                             blog_id=blog_id,
    #                             model_path_name=self.person_model_name)
    #
    # def fashion(self, blog, blog_id):
    #     return self.get_results(blog=blog,
    #                             blog_id=blog_id,
    #                             model_path_name=self.fashion_model_name)
    #
    # def chemical(self, blog, blog_id):
    #     return self.get_results(blog=blog,
    #                             blog_id=blog_id,
    #                             model_path_name=self.chemical_model_name)
    #
    # def disease(self, blog, blog_id):
    #     return self.get_results(blog=blog,
    #                             blog_id=blog_id,
    #                             model_path_name=self.diese_model_name)
    #
    # def organization(self, blog, blog_id):
    #     return self.get_results(blog=blog,
    #                             blog_id=blog_id,
    #                             model_path_name=self.organization_model_name)



