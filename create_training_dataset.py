import csv
import json
import pandas as pd
import numpy as np

def get_experiment_data(filename, data_type='beer'):
    df = pd.read_csv(filename)
    df = df[df['task'] == data_type]
    df = df[df['condition'] =='Conf.+Single']
    print(len(df))
    selected_columns = ['task', 'condition', 'questionId', 'time', 'choice', 'y', 'pred']
    examples = pd.DataFrame()
    for attr in selected_columns:
        examples[attr] = df[attr]
    examples = df.astype({'choice': 'int32',
                          'y': 'int32',
                          'pred': 'int32',
                          'time': 'float64'
                        })
    return examples    

def get_text_and_saliency(filename):
    df = pd.read_json(filename)
    df['questionId'] = np.arange(len(df))
    return df

def combine_data_and_text(experiment_data, text):
    merged = text.merge(experiment_data, left_on='questionId', right_on='questionId')
    print(len(merged))
    return merged

        
experiment_data_beer = get_experiment_data("experiment-data/decision-result-filter.csv", "beer")
text_beer = get_text_and_saliency("task-examples/task-sentiment-beer.json")
merged_data_beer = combine_data_and_text(experiment_data_beer, text_beer)

experiment_data_amzbook = get_experiment_data("experiment-data/decision-result-filter.csv", "amzbook")
text_amzbook = get_text_and_saliency("task-examples/task-sentiment-amzbook.json")
merged_data_amzbook = combine_data_and_text(experiment_data_amzbook, text_amzbook)

merged_data = pd.concat([merged_data_amzbook, merged_data_beer], axis=0)
merged_data.to_csv("examples.csv", index=False)