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
MAX_LENGTH = 512

class SaliencyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        #self.span = [t.index(30522) for t in self.encodings['input_ids']]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        #item['e_span'] = torch.tensor(self.span[idx])
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

def get_local_neighbors(current_text, test_label, tokenizer, sal_tokens=['<sal0>','</sal0>','<sal1>','</sal1>'], no_sal_tokens=['<out_sal>','</out_sal>'], max_n=5):
    word_list = current_text.split(' ')
    neighbors = []
    for n in range(1, max_n + 1):
        for idx, word in enumerate(word_list):
            if idx + 3 * n - 1 >= len(word_list):
                break
            replacement_token_pairs = None
            if word in [sal_tokens[0], sal_tokens[2]]:
                replacement_token_pairs = [no_sal_tokens]
            elif word == no_sal_tokens[0]:
                replacement_token_pairs = [sal_tokens[0:2], sal_tokens[2:]]
            if replacement_token_pairs:
                for cur_pair in replacement_token_pairs:
                    new_word_list = word_list.copy()
                    for i in range(idx, idx + 3 * n, 3):
                        new_word_list[i] = cur_pair[0]
                        new_word_list[i + 2] = cur_pair[1]
                    neighbors.append(' '.join([x for x in new_word_list if x!='[PAD]' and x!='[CLS]' and x!='[SEP]']))
    test_encodings = tokenizer(neighbors, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_dataset = SaliencyDataset(test_encodings, test_label.repeat(len(neighbors)))
    return test_dataset, neighbors

def add_saliency_token(tweets, predicted_class, condition, conf_y, task, sal_tokens=['<sal0>','</sal0>','<sal1>','</sal1>'], no_sal_tokens=['<out_sal>','</out_sal>']):
    new_texts = []

    # median scores from CHI paper
    adaptive_dataset_median_scores = {
        "beer": 0.892,
        "amzbook": 0.889
    }
    for text, pred, cond, conf, data_type in zip(tweets, predicted_class, condition, conf_y, task):
        other = abs(pred - 1)
        # replace string tags of highlighted words based on the condition
        target_start_tag = "<span class=" + "class" + str(pred) + ">"
        other_start_tag = "<span class=" + "class" + str(other) + ">"
        if cond == "Human":
            text = text.replace(target_start_tag, " ")
            text = text.replace(other_start_tag, " ")
        elif cond == "Conf.+Single":
            text = text.replace(target_start_tag, " " + sal_tokens[pred * 2] + " ")
            text = text.replace(other_start_tag, " ")
        elif cond == "Conf.+Adaptive":
            text = text.replace(target_start_tag, " " + sal_tokens[pred * 2] + " ")
            if conf <= adaptive_dataset_median_scores[data_type]:
                text = text.replace(other_start_tag, " " + sal_tokens[other * 2]  + " ")
            else:
                text = text.replace(other_start_tag, " ")
        else:
            text = text.replace(target_start_tag, " " + sal_tokens[pred * 2] + " ")
            text = text.replace(other_start_tag, " " + sal_tokens[other * 2]  + " ")

        # replace existing single end tag with class specific XML end tags
        end_tag = "</span>"
        text = text.replace("\"", "")
        text = text.replace(end_tag, " " + end_tag + " ")
        split_text = text.split()
        next_end_tag = ''
        for idx in range(len(split_text)):
            if split_text[idx] == sal_tokens[0]:
                next_end_tag = sal_tokens[1]
            elif split_text[idx] == sal_tokens[2]:
                next_end_tag = sal_tokens[3]
            elif split_text[idx] == end_tag:
                split_text[idx] = next_end_tag
                next_end_tag = ''

        # join and split text again to get rid of removed end tags
        text = ' '.join([x for x in split_text])
        split_text = text.split()

        # add no_sal tokens around all other tokens
        word_list = []
        start_sal_tokens = [sal_tokens[0], sal_tokens[2]]
        for idx in range(len(split_text)):
            if idx == 0 and split_text[idx] not in sal_tokens + no_sal_tokens:
                word_list.extend([no_sal_tokens[0], split_text[idx], no_sal_tokens[1]])
            elif idx == 0:
                word_list.append(split_text[idx])
            elif split_text[idx] in sal_tokens or split_text[idx - 1] in start_sal_tokens:
                word_list.append(split_text[idx])
            else:
                word_list.extend([no_sal_tokens[0], split_text[idx], no_sal_tokens[1]])
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

def pad_input(input_encodings, pad_val = 0, max_length = MAX_LENGTH):
    return [(np.pad(seq, (0, max_length - len(seq)), 'constant', constant_values=(pad_val))).tolist() for seq in input_encodings]

def create_data_instances(tokenizer):
    tweet_df = pd.read_csv("../full_examples.csv")
    # aggregate samples accross annotators and calculate per-sample accuracy
    tweet_df = aggregate_samples(tweet_df)
    tweet_df.rename(columns={'system':'text'}, inplace=True)

    # train/test/dev split
    data = tweet_df[['text', 'pred_y', 'condition', 'conf_y', 'task', 'percent_correct']]
    train_data, test_data = train_test_split(data, test_size = 0.3, random_state=16)
    test_data, val_data = train_test_split(test_data, test_size = 0.5, random_state=16)
    #print("Mean_score", train_data['percent_correct'].mean())

    train_data_dict, test_data_dict, val_data_dict = {}, {}, {}
    for label in data.columns:
        train_data_dict[label] = train_data[label].values.tolist()
        test_data_dict[label] = test_data[label].values.tolist()
        val_data_dict[label] = val_data[label].values.tolist()

    # tokenize tweets
    train_text = add_saliency_token(train_data_dict['text'], train_data_dict['pred_y'], train_data_dict['condition'], train_data_dict['conf_y'], train_data_dict['task'])
    test_text = add_saliency_token(test_data_dict['text'], test_data_dict['pred_y'], test_data_dict['condition'], test_data_dict['conf_y'], test_data_dict['task'])
    val_text = add_saliency_token(val_data_dict['text'], val_data_dict['pred_y'], val_data_dict['condition'], val_data_dict['conf_y'], val_data_dict['task'])

    train_encodings = tokenizer(train_data_dict['text'], truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_data_dict['text'], truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_data_dict['text'], truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = SaliencyDataset(train_encodings, train_data_dict['percent_correct'])
    test_dataset = SaliencyDataset(test_encodings, test_data_dict['percent_correct'])
    val_dataset = SaliencyDataset(val_encodings, val_data_dict['percent_correct'])

    return train_dataset, test_dataset, val_dataset, test_text


