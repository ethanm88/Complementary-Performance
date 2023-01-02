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


MAX_LENGTH = 512

class SaliencyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, texts, labels, pred_y, Y):
        self.encodings = encodings
        self.texts = texts
        self.labels = labels
        self.pred_y = pred_y
        self.Y = Y

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['texts'] = self.texts[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        item['pred_y'] = self.pred_y[idx]
        item['Y'] = self.Y[idx]
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

def deconstruct_xml(marked_text, sal_dict={'<sal0>': 0,'</sal0>': -1,'<sal1>': 1,'</sal1>': -1}):
    '''
    Deconstructed designations are -1 for unhighlighted, 0 for class 0, and 1 for class 1
    markers are removed
    '''
    designations = []
    labels = sal_dict.values()
    current_label = -1
    for t in marked_text.split(' '):
        if t not in sal_dict:
            designations.append(current_label)
        else:
            current_label = sal_dict[t]
    return designations, labels

def reconstruct_xml(word_list, designations, reverse_sal_dict={0:['<sal0>','</sal0>'], 1:['<sal1>','</sal1>'], -1:[' ',' ']}):
    '''
    Deconstructed designations are -1 for unhighlighted, 0 for class 0, and 1 for class 1
    markers are removed
    '''
    constructed_xml = []
    for i in range(len(designations)):
        if i == 0:
            constructed_xml.append(reverse_sal_dict[designations[i]][0])
        elif designations[i - 1] != designations[i]:
             constructed_xml.extend([reverse_sal_dict[designations[i - 1]][1], reverse_sal_dict[designations[i]][0]])
        constructed_xml.append(word_list[i])
    constructed_xml.append(reverse_sal_dict[designations[-1]][1])
    new_text = ' '.join([x for x in constructed_xml if x!='[PAD]' and x!='[CLS]' and x!='[SEP]' and x != ' '])
    return new_text

def get_local_neighbors(current_text, test_label, tokenizer, sal_tokens=['<sal0>','</sal0>','<sal1>','</sal1>'], max_n=5):
    deconstructed_xml, labels = deconstruct_xml(current_text)
    filtered_word_list = [w for w in current_text.split(' ') if w not in sal_tokens]
    neighbors = []
    for n in range(1, max_n + 1):
        for idx, cur_label in enumerate(deconstructed_xml):
            if idx + n > len(deconstructed_xml):
                break
            cur_xml = deconstructed_xml.copy()
            for l in labels:
                if l != cur_label:
                    cur_xml[idx:idx+n] = [l] * n
                    neighbors.append(reconstruct_xml(filtered_word_list, cur_xml))
    test_encodings = tokenizer(neighbors, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_dataset = SaliencyDataset(test_encodings, test_label.repeat(len(neighbors)), test_label.repeat(len(neighbors)), test_label.repeat(len(neighbors)), test_label.repeat(len(neighbors)))
    return test_dataset, neighbors

def add_saliency_token(tweets, predicted_class, condition, conf_y, task, sal_tokens=['<sal0>','</sal0>','<sal1>','</sal1>']):
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
        elif cond == "Conf.+Adaptive (Expert)":
            target_start_tag_expert = "<span class=" + "\'class" + str(pred) + "\'>"
            other_start_tag_expert = "<span class=" + "\'class" + str(other) + "\'>"
            text = text.replace(target_start_tag_expert, " " + sal_tokens[pred * 2] + " ")
            if conf <= adaptive_dataset_median_scores[data_type]:
                text = text.replace(other_start_tag_expert, " " + sal_tokens[other * 2]  + " ")
            else:
                text = text.replace(other_start_tag_expert, " ")
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

        # combine remove sal tokens to form spans
        if cond == "Conf.+Adaptive (Expert)":
            new_texts.append(text)
        else:
            word_list = []
            reduction_pairs = [[sal_tokens[1], sal_tokens[0]],[sal_tokens[3], sal_tokens[2]]] # marker pairs that can be removed to create a span
            for idx in range(0, len(split_text) - 1):
                if [split_text[idx], split_text[idx + 1]] in reduction_pairs or [split_text[idx - 1], split_text[idx]] in reduction_pairs:
                    continue
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
    for id in tweet_df['testid_modified'].unique():
        row = tweet_df.loc[tweet_df['testid_modified'] == id].iloc[[0]]
        for header in tweet_df.columns:
            avg_dict[header].append(row.loc[list(row.index.values)[0], header])
        current_mean = tweet_df.loc[tweet_df['testid_modified'] == id, 'correctness'].mean()
        avg_dict["percent_correct"].append(current_mean)
    return pd.DataFrame(avg_dict)

def pad_input(input_encodings, pad_val = 0, max_length = MAX_LENGTH):
    return [(np.pad(seq, (0, max_length - len(seq)), 'constant', constant_values=(pad_val))).tolist() for seq in input_encodings]

def create_data_instances(tokenizer):
    set_seed(30)
    tweet_df = pd.read_csv("../full_examples.csv")
    # aggregate samples accross annotators and calculate per-sample accuracy
    tweet_df = aggregate_samples(tweet_df)
    #tweet_df.rename(columns={'system':'text'}, inplace=True)
    tweet_df['text']=np.where(tweet_df['condition'] == "Conf.+Adaptive (Expert)",tweet_df['expert'],tweet_df["system"])
    tweet_df = tweet_df.sort_values(by=['testid_modified'])

    # train/test/dev split
    unique_tweet_ids = tweet_df["testid"].unique()
    train_ids, test_ids = train_test_split(unique_tweet_ids, test_size = 0.3, random_state=16)
    test_ids, val_ids = train_test_split(test_ids, test_size = 0.5, random_state=16)

    data = tweet_df[['text', 'pred_y', 'Y', 'condition', 'conf_y', 'task', 'percent_correct', 'testid']]
    #train_data, test_data = train_test_split(data, test_size = 0.3, random_state=16)
    #test_data, val_data = train_test_split(test_data, test_size = 0.5, random_state=16)
    train_data = data.loc[data['testid'].isin(train_ids)]
    test_data = data.loc[data['testid'].isin(test_ids)]
    val_data = data.loc[data['testid'].isin(val_ids)]

    train_data_dict, test_data_dict, val_data_dict = {}, {}, {}

    for label in data.columns:
        train_data_dict[label] = train_data[label].values.tolist()
        test_data_dict[label] = test_data[label].values.tolist()
        val_data_dict[label] = val_data[label].values.tolist()

    # tokenize tweets
    train_data_dict['text'] = add_saliency_token(train_data_dict['text'], train_data_dict['pred_y'], train_data_dict['condition'], train_data_dict['conf_y'], train_data_dict['task'])
    test_data_dict['text'] = add_saliency_token(test_data_dict['text'], test_data_dict['pred_y'], test_data_dict['condition'], test_data_dict['conf_y'], test_data_dict['task'])
    val_data_dict['text'] = add_saliency_token(val_data_dict['text'], val_data_dict['pred_y'], val_data_dict['condition'], val_data_dict['conf_y'], val_data_dict['task'])

    train_encodings = tokenizer(train_data_dict['text'], truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_data_dict['text'], truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_data_dict['text'], truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = SaliencyDataset(train_encodings, train_data_dict['text'], train_data_dict['percent_correct'], train_data_dict['pred_y'], train_data_dict['Y'])
    test_dataset = SaliencyDataset(test_encodings, test_data_dict['text'], test_data_dict['percent_correct'], test_data_dict['pred_y'], test_data_dict['Y'])
    val_dataset = SaliencyDataset(val_encodings, val_data_dict['text'], val_data_dict['percent_correct'], val_data_dict['pred_y'], val_data_dict['Y'])

    return train_dataset, test_dataset, val_dataset, train_data['percent_correct'].mean()
