from datasets import load_from_disk

data_path = "./resources/data_preprocessed_dataset"
dataset = load_from_disk(data_path)

print(dataset)