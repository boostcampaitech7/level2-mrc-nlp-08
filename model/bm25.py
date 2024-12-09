import os
import pickle
import logging
import numpy as np
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
kiwi=Kiwi()

LOGGER = logging.getLogger()


class BM25Reranker(object):
    def __init__(self, tokenizer=None, bm25_pickle=None):
        self.model = None
        self.tokenizer = tokenizer
        
        if bm25_pickle:
            self._load_bm25_pickle(bm25_pickle)

    def _load_bm25_pickle(self, bm25_pickle):
        print('>>> Loading BM25 model.')
        with open(bm25_pickle, 'rb') as file:
            self.model = pickle.load(file)

    def _save_bm25_pickle(self, model, path):
        if not os.path.exists(path):
            os.makedirs(path)

        pickle_file_path = os.path.join(path, 'bm25_pickle.pkl')
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(model, file)
            
    def _tokenize(self, text):
        # tokenized_text = [self.tokenizer.tokenize(txt) for txt in text]
        tokenized_text = []
        for txt in text:
            try:
                morph_question = kiwi.tokenize(txt)
                morph_question_form = [x.form for x in morph_question if x.tag in ["NNG", "NNP","NNB","NR","NP","SN","SH","SL","VV","VA","XR"]]
                morph_question_form_hf = []
                for x in morph_question_form:
                    morph_question_form_hf.extend(self.tokenizer.tokenize(x))
            except:
                morph_question_form = kiwi.tokenize(txt)
            tokenized_text.append(morph_question_form_hf)
        return tokenized_text

    def _prepend_title_to_text(self, text, title):
        text_with_title = []
        for _text, _title in zip(text, title):
            text_with_title.append(_title + '. ' + _text)
        return text_with_title
        
    def build_bm25_model(self, text, title=None, path='./bm25_model'): 
        if title:
            prepended_text = self._prepend_title_to_text(text, title)
            tokenized_text = self._tokenize(prepended_text) 
        else:
            tokenized_text = self._tokenize(text)
            
        print('>>> Training BM25 model...')
        model = BM25Okapi(tokenized_text) 

        print('>>> Training done.')
        self.model = model
        self._save_bm25_pickle(model, path)

    def get_bm25_rerank_scores(self, questions, doc_ids):
        tokenized_questions = self._tokenize(questions)
        bm25_scores = []
        for question, doc_id in zip(tokenized_questions, doc_ids):
            # bm25_score : [0.        , 0.93729472, 0.        ... ]
            bm25_score = self.model.get_batch_scores(question, doc_id)
            bm25_scores.append(bm25_score)

        return np.array(bm25_scores)