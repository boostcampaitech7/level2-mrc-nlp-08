import pandas as pd
import re
from datasets import Dataset, DatasetDict


def csv_to_datasetdict(csv_file_path, split_name='validation'):
    # CSV 파일 읽기
    #df = pd.read_csv(csv_file_path)
    df = csv_file_path

    def extract_answer(x):
        if not isinstance(x, str):
            return
        text = re.findall(r"'text': array\(\[(['\"]?)(.+?)\1\]", x)[0][1]
        print(text)
        start = re.findall(r"'answer_start': array\(\[(\d+)\]\)", x)
        print(start)
        return {'text': [text], 'answer_start': [int(s) if s else None for s in start]}

    df['answers'] = df['answers'].apply(extract_answer)
    # DataFrame을 Dataset으로 변환
    dataset = Dataset.from_pandas(df)
    
    # Dataset을 DatasetDict에 포함
    dataset_dict = DatasetDict({split_name: dataset})
    
    return dataset_dict

# 사용 예시
csv_file_path = './preprocess_train.csv'
# valcsv_file_path = './preprocess_validation.csv'
# dataset_dict = csv_to_datasetdict(csv_file_path)
df = pd.read_csv(csv_file_path)
df = df.drop("__index_level_0__", axis=1)
df = df.drop("document_id", axis=1)
val_dataset_dict = csv_to_datasetdict(df)
# print(dataset_dict)

# dataset_dict.save_to_disk('./resources/data_preprocessed_dataset')
val_dataset_dict.save_to_disk('./data_preprocessed_dataset_nodocumentid')