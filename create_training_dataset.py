import csv
import json
import pandas as pd
import numpy as np

def get_experiment_data(filename, data_type='beer', condition='Conf.+Single'):
    df = pd.read_csv(filename)
    df = df[df['task'] == data_type]
    df = df[df['condition'] == condition]
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

def combine_data_and_text(experiment_data, text, condition):
    '''
    merge the text data and the annotation results
    include and identifier to differentiate between the different 
    conditions
    '''
    if condition == 'Human':
        identifier = 'H'
    elif condition == "Conf.+Adaptive (Expert)":
        identifier = '0EA'
    else:
        identifier = condition[condition.find('+') + 1]
    merged = text.merge(experiment_data, left_on='questionId', right_on='questionId')
    merged['testid_modified'] = merged['testid'].astype(str) + '_' + identifier
    
    return merged

all_data = []
for dataset_name in ["beer", "amzbook"]:
    for condition in ["Conf.+Single", "Conf.+Double", "Conf.+Adaptive", "Human", "Conf.+Adaptive (Expert)"]:
        experiment_data = get_experiment_data("experiment-data/decision-result-filter.csv", dataset_name, condition)
        text = get_text_and_saliency(f'task-examples/task-sentiment-{dataset_name}.json')
        merged_data = combine_data_and_text(experiment_data, text, condition)
        all_data.append(merged_data)

merged_data = pd.concat(all_data, axis=0)
merged_data.to_csv("full_examples.csv", index=False)