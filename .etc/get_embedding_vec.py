import os
import sys
import torch
import random
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from dense_retrieval import VectorDatabase
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append('../')
from utils.dataloader_dpr import BiEncoderDataset
from utils.utils import init_logging
from model.dpr import Pooler
from model.bm25 import BM25Reranker

LOGGER = logging.getLogger()

seed = 104
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 


def get_tfidf_embedding(args):
    # wiki 데이터 불러오기
    if args.wiki_data_path.split('.')[-1] == 'csv':
        wiki = pd.read_csv(args.wiki_data_path)
        LOGGER.info(f"Load wiki dataset : {len(wiki)}개")
        LOGGER.info(f"Item in wiki {wiki}")
    else:
        raise Exception("wiki 데이터 경로를 다시 확인 해주세요")
    
    # tokenizer 불러오기
    if args.tokenizer == 'kiwi':
        tokenizer = Kiwi()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # context 불러오기
    if args.tokenizer == 'kiwi':
        NotImplemented
    else:
        wiki['title_text'] = wiki['title'] + ' ' + wiki['text']
        contexts = wiki['title_text'].tolist()

    # Vectorizer 불러오고 실행
    tfidfv = TfidfVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 2), max_features=50000)
    p_embedding = tfidfv.fit_transform(contexts)
    
    # 저장
    with open("sparse_embedding_tfidv.bin", "wb") as file:
        pickle.dump(p_embedding, file)
    with open("tfidv.bin", "wb") as file:
        pickle.dump(tfidfv, file)
    LOGGER.info("tf-idf embedding pickle saved")
    
def get_bm25_embedding(args):
    # wiki 데이터 불러오기
    if args.wiki_data_path.split('.')[-1] == 'csv':
        wiki = pd.read_csv(args.wiki_data_path)
        LOGGER.info(f"Load wiki dataset : {len(wiki)}개")
        LOGGER.info(f"Item in wiki {wiki}")
    else:
        raise Exception("wiki 데이터 경로를 다시 확인 해주세요")
    
    # tokenizer 불러오기
    if args.tokenizer == 'kiwi':
        tokenizer = Kiwi()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # context 불러오기
    if args.tokenizer == 'kiwi':
        NotImplemented
    else:
        wiki['title_text'] = wiki['title'] + ' ' + wiki['text']
        contexts = wiki['title_text'].tolist()
        contexts = [tokenizer.tokenize(doc) for doc in tqdm(contexts, desc="BM25 Tokenizing")]
        
    # Vectorizer 불러오고 실행 / bm25 파라미터 자유롭게 변경
    bm25 = BM25Okapi(contexts, k1=1.5, b=0.75)
    with open("bm25.bin", 'wb') as f:
        pickle.dump(bm25, f)

def get_dpr_embedding():
    LOGGER.info('*** Building Vector Database ***')
    args = dense_argument()
    vector_db = VectorDatabase()
    if args.valid_data:
        valid_dataset = BiEncoderDataset.load_valid_dataset(args.valid_data)
        gold_passages = valid_dataset.positive_ctx
    else:
        gold_passages = None
        
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pooler = Pooler(args.pooler)

    vector_db.build_embedding(wiki_path=args.wiki_path,
                              save_path=args.save_path,
                              save_context=args.save_context,
                              tokenizer=tokenizer,
                              embedding_model=model,
                              pooler=pooler,
                              cpu_workers=args.cpu_workers,
                              gold_passages=gold_passages,
                              device = args.device,)
    
    #### Train BM 25 ####
    if args.train_bm25:
        bm25_model = BM25Reranker(tokenizer=tokenizer)
        bm25_model.build_bm25_model(text=vector_db.text,
                                    title=vector_db.title,
                                    path=args.save_path)
    
def main(args):
    init_logging()
    if args.retrieval == 'tf-idf':
        get_tfidf_embedding(args)

    elif args.retrieval == 'bm25':
        get_bm25_embedding(args)

    elif args.retrieval == 'dense':
        get_dpr_embedding()

def dense_argument():
    parser = argparse.ArgumentParser(description='Build vector database with wiki text')
    parser.add_argument('--model', type=str,
                        default="../checkpoint/dpr/context_encoder",
                        help='Directory of pretrained encoder model'
                       )
    parser.add_argument('--wiki_path', type=str, default='../resources/processed/modified_wikipedia_documents.csv',
                        help='csv 형식, 컬럼에 document id, title, context가 존재해야 함'
                       )
    parser.add_argument('--valid_data', type=str,
                        default='../resources/dpr/dpr_valid_v2.json',
                        help='Path of validation dataset'
                       )
    parser.add_argument('--save_path', type=str, 
                        default='./pickles_kiwi_main_hf_dpr2',
                        help='Save directory of faiss index'
                       )
    parser.add_argument('--save_context', action='store_true', 
                        default=True,
                        help='Save text and title with faiss index'
                       )
    parser.add_argument('--train_bm25', action='store_true', 
                        default=True,
                        help='Train bm25 with the same corpus'
                       )
    parser.add_argument('--num_sent', type=int, 
                        default=5,
                        help='Number of sentences consisting of a wiki chunk'
                       )
    parser.add_argument('--overlap', type=int, 
                        default=0,
                        help='Number of overlapping sentences between consecutive chunks'
                       )
    parser.add_argument('--pooler', default='cls', type=str,
                        help='Pooler type : {pooler_output|cls|mean|max}'
                       )       
    parser.add_argument('--max_length', type=int, default=512,
                        help='Max length for encoder model'
                       )
    parser.add_argument('--batch_size', type=int, 
                        default=128,
                        help='Batch size'
                       )
    parser.add_argument('--cpu_workers', type=int, 
                        default=os.cpu_count()//2,
                        required=False,
                        help='Number of cpu cores used in chunking wiki text'
                       ) 
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    parser.add_argument('--random_seed', default = 104, type=int,
                        help = 'Random seed'
                       ) 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=str, default="dense", help="tfidf, bm25, dense")
    parser.add_argument("--wiki_data_path", type=str, default="../resources/wiki/remove_duplicates.csv")
    parser.add_argument("--tokenizer", type=str, default="monologg/kobigbird-bert-base", help="huggingface 모델명 혹은 kiwi(한국어 형태소 분석기) 선택")    
    args = parser.parse_args()
    
    main(args)