import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_goemotions():
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions")
    label_names = dataset['train'].features['labels'].feature.names
    return dataset, label_names

def encode_labels(example, label_names):
    vector = [0] * len(label_names)
    for label in example['labels']:
        vector[label] = 1
    example['label_vector'] = vector
    return example

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    dataset, label_names = load_goemotions()
    dataset = dataset.map(lambda x: encode_labels(x, label_names))

    df_train = pd.DataFrame({'text': dataset['train']['text'], 'labels': dataset['train']['label_vector']})
    df_val = pd.DataFrame({'text': dataset['validation']['text'], 'labels': dataset['validation']['label_vector']})
    df_test = pd.DataFrame({'text': dataset['test']['text'], 'labels': dataset['test']['label_vector']})

    save_to_csv(df_train, "../data/processed/train.csv")
    save_to_csv(df_val, "../data/processed/val.csv")
    save_to_csv(df_test, "../data/processed/test.csv")
