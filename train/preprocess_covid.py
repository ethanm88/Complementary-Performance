import csv
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import re

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

set_seed(30)
MAX_LENGTH = 128

class SaliencyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.span = [t.index(30522) for t in self.encodings['input_ids']]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['e_span'] = torch.tensor(self.span[idx])
        return item

    def __len__(self):
        return len(self.labels)

def reconstruct_text_tokens(input_ids, tokenizer):
    word_list = []
    for id in input_ids.squeeze().tolist():
        word_list.append(tokenizer.decode([id]))
    return word_list

def reconstruct_text(input_ids, tokenizer):
    word_list = reconstruct_text_tokens(input_ids, tokenizer)
    text = ' '.join([x for x in word_list if x!='[PAD]' and x!='[CLS]' and x!='[SEP]'])
    text = text.replace(" ##","")
    return text

def get_local_neighbors(current_text, test_label, tokenizer, sal_token='<in_sal>', no_sal_token='<out_sal>'):
    word_list = current_text.split(' ')
    neighbors = []
    for idx, word in enumerate(word_list):
        new_word_list = None
        if word == sal_token:
            new_word_list = word_list.copy()
            new_word_list[idx] = no_sal_token
        elif word == no_sal_token:
            new_word_list = word_list.copy()
            new_word_list[idx] = sal_token
        if new_word_list:
            new_text = ' '.join([x for x in new_word_list if x!='[PAD]' and x!='[CLS]' and x!='[SEP]'])
            neighbors.append(new_text)
    #print('neighbors', neighbors)
    #neighbors = neighbors[0:5]
    test_encodings = tokenizer(neighbors, truncation=True, padding=True, max_length=128)
    test_encodings['input_ids'] = pad_input(test_encodings['input_ids'])
    test_encodings['token_type_ids'] = pad_input(test_encodings['token_type_ids'])
    test_encodings['attention_mask'] = pad_input(test_encodings['attention_mask'])
    test_dataset = SaliencyDataset(test_encodings, test_label.repeat(len(neighbors)))
    return test_dataset, neighbors

def add_saliency_token(tweets, saliency, cp_spans, sal_token='<in_sal>', no_sal_token='<out_sal>', cp_token='<cp>'):
    new_texts = []
    for text, sal, cp_span in zip(tweets, saliency, cp_spans):
        word_list = []
        for idx, token in enumerate(text.split()):
            if idx >= cp_span[0] and idx < cp_span[1]:
                word_list.append(cp_token)
            else:
                if sal[idx] == 1:
                    word_list.append(sal_token)
                else:
                    word_list.append(no_sal_token)
            word_list.append(token)

        new_text = ' '.join([x for x in word_list])
        new_texts.append(new_text)
    return new_texts
        
        

def create_data_instances(tokenizer):
    tweet_df = pd.read_csv("../initial_data.csv")
    # create accuracy values
    tweet_df["gold_annotation"] = np.where(tweet_df['gold_annotation'].gt(3), 1, 0)
    tweet_df["annotation"] = np.where(tweet_df['annotation'].gt(3), 1, 0)
    tweet_df['correctness'] = np.where(tweet_df['annotation'] == tweet_df['gold_annotation'], 1, 0)
    #print(tweet_df[['gold_annotation','annotation','correctness']])
    tweet_df['saliency_scores'] = tweet_df['saliency_scores'].str.split()
    tweet_df['cp_span'] = tweet_df['cp_span'].str.split()
    labels = tweet_df['correctness']
    
    data = tweet_df.drop(columns=["ids", "time", "annotation", "gold_annotation", "correctness"])

    # split data
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size = 0.2, random_state=15)

    train_label, test_label = train_label.values, test_label.values
    train_text, test_text = train_data['tweet'].values.tolist(), test_data['tweet'].values.tolist()
    train_saliency, test_saliency = train_data['saliency_scores'].values.tolist(), test_data['saliency_scores'].values.tolist()
    train_cp_span, test_cp_span = train_data['cp_span'].values.tolist(), test_data['cp_span'].values.tolist()

    train_saliency = [[(float)(sal[i]) != 0.0  for i in range(len(sal))] for sal in train_saliency]
    test_saliency = [[(float)(sal[i]) != 0.0 for i in range(len(sal))] for sal in test_saliency]
    train_cp_span  = [[(int)(idx) for idx in cp_span] for cp_span in train_cp_span]
    test_cp_span  = [[(int)(idx) for idx in cp_span] for cp_span in test_cp_span]

    # tokenize tweets
    train_text = add_saliency_token(train_text, train_saliency, train_cp_span)
    test_text = add_saliency_token(test_text, test_saliency, test_cp_span)

    train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_text, truncation=True, padding=True, max_length=128)

    train_encodings['input_ids'] = pad_input(train_encodings['input_ids'])
    test_encodings['input_ids'] = pad_input(test_encodings['input_ids'])

    train_encodings['token_type_ids'] = pad_input(train_encodings['token_type_ids'])
    test_encodings['token_type_ids'] = pad_input(test_encodings['token_type_ids'])

    train_encodings['attention_mask'] = pad_input(train_encodings['attention_mask'])
    test_encodings['attention_mask'] = pad_input(test_encodings['attention_mask'])

    train_dataset = SaliencyDataset(train_encodings, train_label)
    test_dataset = SaliencyDataset(test_encodings, test_label)

    return train_dataset, test_dataset, test_text

def pad_input(input_encodings, pad_val = 0, max_length = MAX_LENGTH):
    return [(np.pad(seq, (0, 128 - len(seq)), 'constant', constant_values=(pad_val))).tolist() for seq in input_encodings]


