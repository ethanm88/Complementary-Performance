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

def add_saliency_token(tweets, predicted_class, sal_token='<in_sal>', no_sal_token='<out_sal>'):
    new_texts = []
    for text, pred in zip(tweets, predicted_class):
        target_start_tag = "<span class=" + "class" + str(pred) + ">"
        other_start_tag = "<span class=" + "class" + str(abs(pred - 1)) + ">"
        end_tag = "</span>"
        text = text.replace(target_start_tag, sal_token + " ")
        text = text.replace(other_start_tag, no_sal_token  + " ")
        text = text.replace(end_tag, "")
        text = text.replace("\"", "")
        word_list = []
        split_text = text.split()
        for idx in range(len(split_text)):
            if idx == 0 and split_text[idx] not in [sal_token, no_sal_token]:
                word_list.append(no_sal_token)
                word_list.append(split_text[idx])
            elif idx == 0:
                word_list.append(split_text[idx])
            elif split_text[idx] in [sal_token, no_sal_token]:
                word_list.append(split_text[idx])
            elif split_text[idx - 1] in [sal_token, no_sal_token]:
                word_list.append(split_text[idx])
            else:
                word_list.append(no_sal_token)
                word_list.append(split_text[idx])
        new_text = ' '.join([x for x in word_list])
        new_texts.append(new_text)
    return new_texts
        
def aggregate_samples(tweet_df):
    tweet_df['correctness'] = np.where(tweet_df['y'] == tweet_df['choice'], 1, 0)
    avg_dict = {}
    for header in tweet_df.columns:
        avg_dict[header] = []
    avg_dict["percent_correct"] = []
    for id in tweet_df['testid'].unique():
        row = tweet_df.loc[tweet_df['testid'] == id].iloc[[0]]
        for header in tweet_df.columns:
            avg_dict[header].append(row.loc[list(row.index.values)[0], header])
        current_mean = tweet_df.loc[tweet_df['testid'] == id, 'correctness'].mean()
        avg_dict["percent_correct"].append(current_mean)
    return pd.DataFrame(avg_dict)

def create_data_instances(tokenizer):
    tweet_df = pd.read_csv("../examples.csv")
    # aggregate samples accross annotators and calculate per-sample accuracy
    tweet_df = aggregate_samples(tweet_df)
    
    # train/test/dev split
    y_true = tweet_df['percent_correct']
    data = tweet_df[['system', 'pred_y']]
    train_data, test_data, train_y_true, test_y_true = train_test_split(data, y_true, test_size = 0.2, random_state=8)
    test_data, val_data, test_y_true, val_y_true = train_test_split(test_data, test_y_true, test_size = 0.5, random_state=48)

    train_text, test_text, val_text = train_data['system'].values.tolist(), test_data['system'].values.tolist(), val_data['system'].values.tolist()
    train_y_true, test_y_true, val_y_true = train_y_true.values, test_y_true.values, val_y_true.values
    train_pred_class, test_pred_class, val_pred_class = train_data['pred_y'].values.tolist(), test_data['pred_y'].values.tolist(), val_data['pred_y'].values.tolist()


    # tokenize tweets
    train_text = add_saliency_token(train_text, train_pred_class)
    test_text = add_saliency_token(test_text, test_pred_class)
    val_text = add_saliency_token(val_text, val_pred_class)

    train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(test_text, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_text, truncation=True, padding=True, max_length=256)

    #train_encodings['input_ids'] = pad_input(train_encodings['input_ids'])
    #test_encodings['input_ids'] = pad_input(test_encodings['input_ids'])

    #train_encodings['token_type_ids'] = pad_input(train_encodings['token_type_ids'])
    #test_encodings['token_type_ids'] = pad_input(test_encodings['token_type_ids'])

    #train_encodings['attention_mask'] = pad_input(train_encodings['attention_mask'])
    #test_encodings['attention_mask'] = pad_input(test_encodings['attention_mask'])

    train_dataset = SaliencyDataset(train_encodings, train_y_true)
    test_dataset = SaliencyDataset(test_encodings, test_y_true)
    val_dataset = SaliencyDataset(val_encodings, val_y_true)

    return train_dataset, test_dataset, val_dataset, test_text

def pad_input(input_encodings, pad_val = 0, max_length = MAX_LENGTH):
    return [(np.pad(seq, (0, 128 - len(seq)), 'constant', constant_values=(pad_val))).tolist() for seq in input_encodings]


'''
    train_data, test_data, val_data = data[data['testid'].isin(train_ids)]['system'], data[data['testid'].isin(test_ids)]['system'], data[data['testid'].isin(val_ids)]['system']

    train_label, test_label, val_label = labels[labels['testid'].isin(train_ids)]['correctness'], labels[labels['testid'].isin(test_ids)]['correctness'], labels[labels['testid'].isin(val_ids)]['correctness']
 
    train_text, test_text, val_text = train_data.values.tolist(), test_data.values.tolist(), val_data.values.tolist()
    train_label, test_label, val_label = train_label.values, test_label.values, val_label.values

    zipped_train = list(zip(train_text, train_label))
    zipped_test = list(zip(test_text, test_label))
    zipped_val = list(zip(val_text, val_label))

    random.shuffle(zipped_train)
    random.shuffle(zipped_test)
    random.shuffle(zipped_val)

    train_text, train_label = zip(*zipped_train)
    test_text, test_label = zip(*zipped_test)
    val_text, val_label = zip(*zipped_val)

    train_text, train_label = list(train_text), list(train_label)
    test_text, test_label = list(test_text), list(test_label)
    val_text, val_label = list(val_text), list(val_label)


'''