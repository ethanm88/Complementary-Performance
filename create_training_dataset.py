import csv
import json
import pandas as pd
import numpy as np

def get_experiment_data(filename, data_type='amzbook'):
    df = pd.read_csv(filename)
    df = df[df['task'] == data_type]
    df = df[df['condition'] =='Conf.+Single']
    selected_columns = ['task', 'condition', 'questionId', 'time', 'choice', 'y', 'pred']
    examples = pd.DataFrame()
    for attr in selected_columns:
        examples[attr] = df[attr]
    return examples    

def get_text_and_saliency(filename):
    df = pd.read_json(filename)
    df['questionId'] = np.arange(len(df))
    return df

def combine_data_and_text(experiment_data, text):
    merged = text.merge(experiment_data, left_on='questionId', right_on='questionId')
    return merged

        
experiment_data = get_experiment_data("experiment-data/decision-result-filter.csv")
text = get_text_and_saliency("task-examples/task-sentiment-amzbook.json")
merged_data = combine_data_and_text(experiment_data, text)
merged_data.to_csv("examples.csv", index=False)