import torch
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from model.dpr import Pooler
from utils.dataloader_dpr import BiEncoderDataset
from database.dense_retrieval import VectorDatabase
from model.bm25 import BM25Reranker
from utils.metric import get_topk_document
from utils.utils import init_logging
from utils.dataloader_reader import load_from_disk

LOGGER = logging.getLogger()


def search_evaluation(q_encoder, tokenizer, test_dataset, faiss_index, text_index, text, title, 
                      search_k=2000, bm25_model=None, faiss_weight=1, bm25_weight=0.5, max_length=512, 
                      pooler=None, padding=True, truncation=True, batch_size=32, device='cuda'):
    
    datasets = load_from_disk("./resources/data/test_dataset")
    print(f">>> Text dataset size: {datasets}")
    df = datasets['validation'].to_pandas()
    print(df)

    question = df['question'].tolist()
    question_idx = df['id'].tolist()
    
    # BM25를 통해 각 question에 해당하는 corpus 유사도 점수 계산
    print('>>> Searching documents using BM25 index.')
    question_test_idx = np.array([text_index for _ in range(len(question))])
    
    bm25_scores = bm25_model.get_bm25_rerank_scores(question, question_test_idx)
    # shape : (600, 138564) = (valid 질문 수, passage 수)
    with open("inference_bm25_kiwi_main-hf_scores.pkl", 'wb') as file:
        pickle.dump(bm25_scores, file)
    
    with open("inference_bm25_kiwi_main-hf_scores.pkl", 'rb') as file:
        bm25_scores = pickle.load(file)
        print("Load bm25 scores:", bm25_scores.shape)
    
    for idx in range(bm25_scores.shape[0]):
        sorted_idx = np.argsort(bm25_scores[idx])[::-1]
        question_test_idx[idx] = question_test_idx[idx][sorted_idx]
    
    # print('>>> Save documents using BM25')
    # get_topk_document(question_test_idx, question, question_idx, text_index, text, title, "./resources/retrieval/top40_bm25_retrieval_dataset2")

    # Question Embedding by DPR Model.
    q_encoder = q_encoder.to(device)
    q_encoder.eval()
    question_embed = []
    for start_index in tqdm(range(0, len(question), batch_size)):
        batch_question = question[start_index : start_index + batch_size]
           
        q_batch = tokenizer(batch_question,
                            padding=padding,
                            max_length=max_length,
                            truncation=truncation,)
        
        q_encoder.eval()
        with torch.no_grad():
            q_output = q_encoder(input_ids=torch.tensor(q_batch['input_ids']).to(device),
                                 attention_mask=torch.tensor(q_batch['attention_mask']).to(device),
                                 token_type_ids=torch.tensor(q_batch['token_type_ids']).to(device),)
        
        attention_mask = torch.tensor(q_batch['attention_mask'])
        
        if pooler:
            pooler_output = pooler(attention_mask, q_output).cpu()
        else:
            pooler_output = q_output.last_hidden_state[:,0,:].cpu()
        
        question_embed.append(pooler_output)
     
    question_embed = np.vstack(question_embed) 

    D, I = faiss_index.search(question_embed, search_k) # I-faiss index: (question num * k), dpr_valid_v2 shape : (240, 2000)

    print('>>> Reranking : DPR -> BM25')
    bm25_scores = bm25_model.get_bm25_rerank_scores(question, I)
    dpr_bm25_scores = faiss_weight * D + bm25_weight * bm25_scores
    for idx in range(dpr_bm25_scores.shape[0]):
        sorted_idx = np.argsort(dpr_bm25_scores[idx])[::-1]
        # D[idx] = D[idx][sorted_idx]
        I[idx] = I[idx][sorted_idx]
    
    print('>>> Save documents using DPR -> BM25')
    get_topk_document(I, question, question_idx, text_index, text, title, "./resources/retrieval/top40_dpr2+bm25_retrieval_dataset")
    

def main(args):
    init_logging()
    
    LOGGER.info('*** Top-k Retrieval Accuracy ***')
    
    # Load model & tokenizer
    q_encoder = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    pooler = Pooler(args.pooler)
    
    # Load valid dataset.
    test_dataset = BiEncoderDataset.load_valid_dataset(args.valid_data)

    # Load faiss index & context
    faiss_vector = VectorDatabase(args.faiss_path, args.context_path)
    
    faiss_index = faiss_vector.faiss_index
    text_index = faiss_vector.text_index
    text = faiss_vector.text
    # print(text[:5])
    title = faiss_vector.title
    # print(title[:5])
    # Load bm25 model.
    if args.bm25_path:
        bm25_model = BM25Reranker(tokenizer=tokenizer, bm25_pickle=args.bm25_path)
    else:
        bm25_model = None
    
    # Get top-k accuracy
    search_evaluation(q_encoder, tokenizer, test_dataset, faiss_index, text_index, text, title, search_k=args.search_k,
                               bm25_model=bm25_model, faiss_weight=args.faiss_weight, bm25_weight=args.bm25_weight,
                               max_length=args.max_length, pooler=pooler, padding=args.padding, truncation=args.truncation,
                               batch_size=args.batch_size, device=args.device)


def argument_parser():
    parser = argparse.ArgumentParser(description='get topk-accuracy of retrieval model')
    parser.add_argument('--model', type=str, 
                        # default = './checkpoint/jhgan-ko-sroberta-multitask/question_encoder',
                        default = './checkpoint/dpr/question_encoder',
                        help='Directory of pretrained encoder model'
                       )
    parser.add_argument('--valid_data', type=str,
                        default='./resources/dpr/dpr_valid_v2.json',
                        help='Path of validation dataset'
                       )
    parser.add_argument('--faiss_path', type=str,
                        # default='./database/pickles_kiwi_main_hf/faiss_pickle.pkl',
                        default='./database/pickles_kiwi_main_hf_dpr2/faiss_pickle.pkl',
                        help='Path of faiss pickle'
                       )
    parser.add_argument('--bm25_path', type=str,
                        # default='./database/pickles_kiwi_main_hf/faiss_pickle.pkl',
                        default='./database/pickles_kiwi_main_hf_dpr2/bm25_pickle.pkl',
                        help='Path of BM25 Model'
                       )
    parser.add_argument('--context_path', type=str,
                            default='./database/pickles_kiwi_main_hf_dpr2/context_pickle.pkl',
                            help='Path of BM25 Model'
                        )
    parser.add_argument('--faiss_weight', default=0.8, type=float, 
                        help='Weight for semantic search'
                       )
    parser.add_argument('--bm25_weight', default=0.2, type=float, 
                        help='Weight for BM25 rerank score'
                       )
    parser.add_argument('--search_k', default=2000, type=int,
                        help='Number of retrieved documents'
                       )    
    parser.add_argument('--max_length', default=512, type=int,
                        help='Max length of sequence'
                       )                        
    parser.add_argument('--pooler', default='cls', type=str,
                        help='Pooler type : {pooler_output|cls|mean|max}'
                       )
    parser.add_argument('--padding', action="store_false", default=True,
                        help='Add padding to short sentences'
                       )
    parser.add_argument('--truncation', action="store_false", default=True,
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size'
                       )
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    main(args)