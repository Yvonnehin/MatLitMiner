import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import psie
import os
import json

BERT_VERSION = r'./pretrained_models/m3rg-iitd/matscibert'
MAX_LEN = 256
extr_target = 'Solvus'
MAIN_DIR = os.getcwd()
MODEL_DIR = os.path.join("models", extr_target, "classifier")
CORPUS = os.path.join("corpus", extr_target, "classifier/corpus_sentences.json")
OUTPUT = "relevant_sentences"
device = "cuda" if torch.cuda.is_available() else "cpu"

from datasets import load_dataset
model = psie.classifier.BertClassifier()
model.load_state_dict(torch.load('./classifier.pt'), strict=False)
model.to(device)

# INPUT_DIR = ...
# OUTPUT_FILE = ...

combined_filtered_sentences = {"sentence": [], "source": []}
tokenizer = BertTokenizerFast.from_pretrained(BERT_VERSION)

def encode(paper):
    return tokenizer(paper["sentence"], truncation=True, max_length=MAX_LEN, padding="max_length")

def process_batch(files):
    batch_filtered_sentences = {"sentence": [], "source": []}
    
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(INPUT_DIR, filename)
            
            try:
                dataset = load_dataset('json', data_files=file_path)['train']
                dataset = dataset.map(encode, batched=True)
                dataset.set_format(type="torch", columns=["source", "sentence", "input_ids", "attention_mask"])
                
                dataset_loader = DataLoader(dataset, batch_size=32, shuffle=False)
                pred = model.predict(dataset_loader, device)
                
                predictions = []
                for i in range(len(pred)):
                    predictions.append(np.argmax(pred[i].cpu().numpy())) 
                
                for i in range(len(predictions)):
                    if predictions[i] == 1:
                        batch_filtered_sentences["sentence"].append(dataset[i]["sentence"])
                        batch_filtered_sentences["source"].append(dataset[i]["source"])
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    return batch_filtered_sentences

batch_size = 1000  # 每次处理1000个文件
all_files = os.listdir(INPUT_DIR)
num_batches = (len(all_files) + batch_size - 1) // batch_size

for batch_num in range(num_batches):
    batch_files = all_files[batch_num * batch_size:(batch_num + 1) * batch_size]
    batch_filtered_sentences = process_batch(batch_files)
    
    combined_filtered_sentences["sentence"].extend(batch_filtered_sentences["sentence"])
    combined_filtered_sentences["source"].extend(batch_filtered_sentences["source"])
    
    print(f"Batch {batch_num + 1}/{num_batches} finished")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
    json.dump(combined_filtered_sentences, file, ensure_ascii=False, indent=4)

print("处理完成")