import os
import json
import asyncio
import pickle
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from nltk import sent_tokenize
from datasets import load_from_disk

class VectorDatabase(object):
    
    def __init__(self, faiss_pickle=None, context_pickle=None):
        self.text = None
        self.title = None
        self.text_index = None
        self.faiss_index = None
        
        if faiss_pickle:
            self._load_faiss_pickle(faiss_pickle)

        if context_pickle:
            self._load_context_pickle(context_pickle)
        
    def _load_faiss_pickle(self, faiss_pickle):
        print('>>> Loading faiss index.')
        with open(faiss_pickle, 'rb') as file:
            data = pickle.load(file)
            self.text_index = data.get('text_index', None)
            self.faiss_index = data.get('faiss_index', None)

    def _load_context_pickle(self, context_pickle):
        print('>>> Loading text and title.')
        with open(context_pickle, 'rb') as file:
            data = pickle.load(file)
            self.text = data.get('text', None)
            self.title = data.get('title', None) 
        
    def _chunk_context(self, context, title, num_sents, overlaps):
        _txt_lst, _title_lst = [], []
        start, end = 0, num_sents
        total_sents = sent_tokenize(context)
        
        while start < len(total_sents):
            chunk = total_sents[start:end]
            _txt_lst.append(' '.join(chunk))
            _title_lst.append(title)
            
            start += (num_sents - overlaps)
            end = start + num_sents

        return _txt_lst, _title_lst
    
    def _load_wikidata_by_chunk(self, wiki_path, num_sent=5, overlap=0, cpu_workers=None, gold_passages=None):
        print('>>> Loading wiki data.')
        df = pd.read_csv(wiki_path)
        print(df)
        
        # Store text from test set first.
        idx_lst, txt_lst, title_lst = [], [], [] 
        if gold_passages:
            for ctx in gold_passages:
                if ctx['idx'] not in idx_lst:
                    idx_lst.append(ctx['idx'])
                    txt_lst.append(ctx['text'])
                    title_lst.append(ctx['title'].replace('_', ' '))
        
        print('>>> Parsing and chunking wiki data.')
        for idx in tqdm(range(len(df))):
            sample = df.loc[idx]
            title = sample['title']
            text = sample['text']
            _txt_lst, _title_lst = self._chunk_context(text, title, num_sent, overlap)
            txt_lst.extend(_txt_lst)
            title_lst.extend(_title_lst)
            
        for _idx in range(idx_lst[-1]+1, len(txt_lst)):
            idx_lst.append(_idx)

        print(f'>>> Total number of passages: ')

        return idx_lst, txt_lst, title_lst

    def _load_wikidata(self, wiki_path, num_sent=5, overlap=0, cpu_workers=None, gold_passages=None):
        print('>>> Loading valid data.')
        datasets = load_from_disk("../resources/data/train_dataset")
        df_valid = datasets['validation'].to_pandas()
        print('>>> Loading wiki data.')
        df = pd.read_csv(wiki_path)
        
        # Store text from test set first.
        idx_lst, txt_lst, title_lst = [], [], [] 
        for idx, row in df_valid.iterrows():
            idx_lst.append(idx)
            
            title = row['title'].strip()
            title_lst.append(title)
            
            txt = row['context'].strip()
            txt_lst.append(txt)
        
        for _, row in df.iterrows():
            title = row['title'].strip()
            title_lst.append(title)
            
            txt = row['text'].strip()
            txt_lst.append(txt)
            
        for _idx in range(idx_lst[-1]+1, len(txt_lst)):
            idx_lst.append(_idx)
        print(idx_lst[239:242])
        print(len(idx_lst))
        print(f'>>> Total number of passages: ')
        return idx_lst, txt_lst, title_lst
    
    def encode_text(self, title, text, embedding_model, tokenizer, pooler=None, max_length=512, batch_size=32, device='cuda'):
    
        print('>>> Encoding wiki data.')
        embedding_model = embedding_model.to(device)
        
        all_ctx_embed = []

        embedding_model.eval()
        for start_index in tqdm(range(0, len(text), batch_size)):
            batch_txt = text[start_index : start_index + batch_size]
            batch_title = title[start_index : start_index + batch_size]
                           
            batch = tokenizer(batch_title,
                              batch_txt,
                              padding=True,
                              truncation=True,
                              max_length=max_length,)

            with torch.no_grad():
                output = embedding_model(input_ids=torch.tensor(batch['input_ids']).to(device),
                                        attention_mask=torch.tensor(batch['attention_mask']).to(device),
                                        token_type_ids=torch.tensor(batch['token_type_ids']).to(device),)

                attention_mask = torch.tensor(batch['attention_mask']).to(device)
                
                if pooler:
                    pooler_output = pooler(attention_mask, output) 
                else:
                    pooler_output = output.last_hidden_state[:,0,:]
        
            all_ctx_embed.append(pooler_output.cpu())

        all_ctx_embed = np.vstack(all_ctx_embed) 
        
        return all_ctx_embed
    
    def build_embedding_chunk(self,
                        wiki_path=None,
                        save_path=None,
                        save_context=None,
                        tokenizer=None,
                        embedding_model=None,
                        pooler = None,
                        num_sent=5,
                        overlap=0,
                        cpu_workers=None,
                        gold_passages=None,
                        embedding_size=768,
                        max_length=512,
                        batch_size=32,
                        device='cuda',
                        ):
        
        idx_lst, txt_lst, title_lst = self._load_wikidata_by_chunk(wiki_path, num_sent, overlap, cpu_workers, gold_passages)

        all_embeddings = self.encode_text(title_lst, txt_lst, embedding_model, tokenizer, pooler, max_length, batch_size, device)
        
        faiss.normalize_L2(all_embeddings)
        faiss_index = faiss.IndexFlatIP(embedding_size)
        faiss_index.add(all_embeddings)
        
        print(">>> Saving faiss pickle. It contains \'text_index\' and \'faiss_index\'.")  
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pickle_file_path = os.path.join(save_path, 'faiss_pickle.pkl')
    
        with open(pickle_file_path, 'wb') as file:
            pickle.dump({
                'text_index': idx_lst,
                'faiss_index':faiss_index,
            }, file)

        if save_context:
            print(">>> Saving context pickle. It contains \'title\' and \'text\'.")
            pickle_file_path = os.path.join(save_path, 'context_pickle.pkl')
            with open(pickle_file_path, 'wb') as file:
                pickle.dump({
                    'title': title_lst,
                    'text':txt_lst,
                }, file)
                    
        self.text = txt_lst
        self.title = title_lst
        self.text_index = idx_lst
        self.faiss_index = faiss_index

        print(f'>>> Total number of passages: {len(self.text_index)}')

    def build_embedding(self,
                        wiki_path=None,
                        save_path=None,
                        save_context=None,
                        tokenizer=None,
                        embedding_model=None,
                        pooler = None,
                        num_sent=5,
                        overlap=0,
                        cpu_workers=None,
                        gold_passages=None,
                        embedding_size=768,
                        max_length=512,
                        batch_size=32,
                        device='cuda',
                        ):
        
        idx_lst, txt_lst, title_lst = self._load_wikidata(wiki_path, num_sent, overlap, cpu_workers, gold_passages)

        all_embeddings = self.encode_text(title_lst, txt_lst, embedding_model, tokenizer, pooler, max_length, batch_size, device)
        
        faiss.normalize_L2(all_embeddings)
        faiss_index = faiss.IndexFlatIP(embedding_size)
        faiss_index.add(all_embeddings)
        
        print(">>> Saving faiss pickle. It contains \'text_index\' and \'faiss_index\'.")  
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pickle_file_path = os.path.join(save_path, 'faiss_pickle.pkl')
    
        with open(pickle_file_path, 'wb') as file:
            pickle.dump({
                'text_index': idx_lst,
                'faiss_index':faiss_index,
            }, file)

        if save_context:
            print(">>> Saving context pickle. It contains \'title\' and \'text\'.")
            pickle_file_path = os.path.join(save_path, 'context_pickle.pkl')
            with open(pickle_file_path, 'wb') as file:
                pickle.dump({
                    'title': title_lst,
                    'text':txt_lst,
                }, file)
                    
        self.text = txt_lst
        self.title = title_lst
        self.text_index = idx_lst
        self.faiss_index = faiss_index

        print(f'>>> Total number of passages: {len(self.text_index)}')