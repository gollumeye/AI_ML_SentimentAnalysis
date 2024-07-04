import json
import re
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
from dataPreparation.contractions import contractions
from dataPreparation.feature_engineering_baselines import get_context_embedding
import random

def preprocess_surveys(text_example):
    #text_example = text_example.lower()

    # handle contractions like I'll etc.
    words = text_example.split()
    text_example = [contractions[word] if word in contractions else word for word in words]
    text_example = ' '.join(text_example)

    #text_example = re.sub(r'[^\w\s]', '', text_example)  # remove punctuation

    return text_example


def preprocess_tweets(text_example):
    text_example = re.sub(r'@\w+', '', text_example) #remove usernames
    text_example = re.sub(r'#\w+', '', text_example) #remove hashtags
    text_example = re.sub(r'\d+', '', text_example) #remove numbers

    return text_example

def get_survey_data_for_bert(num_texts_per_label):
    with open('surveys.json', 'r') as file:
        data = json.load(file)

    random.shuffle(data)

    texts = []
    labels = []
    count_per_label = {'positive': 0, 'neutral': 0, 'negative': 0}

    for entry in data:
        label = entry['label']
        if count_per_label[label] < num_texts_per_label:
            texts.append(entry['text'])
            labels.append(label)
            count_per_label[label] += 1

    print("Number of instances per label:")
    for label, count in count_per_label.items():
        print(f"{label}: {count}")

    preprocessed_texts = [preprocess_surveys(text) for text in texts]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = tokenizer(preprocessed_texts, padding=True, truncation=True, return_tensors='pt') #pt -> return pytorch tensor
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2} #bert needs numerical labels
    numerical_labels = [label_map[label] for label in labels]

    return tokenized_texts, numerical_labels


def get_tweet_data_for_bert(num_texts_per_label):
    df = pd.read_csv('tweets.csv')

    texts = []
    labels = []
    count_per_label = {'positive': 0, 'neutral': 0, 'negative': 0}
    for _, row in df.iterrows():
        label = row['sentiment']
        if count_per_label[label] < num_texts_per_label:
            texts.append(row['Text'])
            labels.append(label)
            count_per_label[label] += 1

    print("Number of instances per label:")
    for label, count in count_per_label.items():
        print(f"{label}: {count}")

    preprocessed_texts = [preprocess_tweets(text) for text in texts]

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    tokenized_texts = tokenizer(preprocessed_texts, padding='max_length', truncation=True, max_length=128,
                                return_tensors='pt')
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    numerical_labels = [label_map[label] for label in labels]

    return tokenized_texts, numerical_labels

def get_survey_data_for_baselines(number_of_examples):
    with open('surveys.json', 'r') as file:
        data = json.load(file)

    random.shuffle(data)

    X_positive = []
    y_positive = []
    X_negative = []
    y_negative = []
    X_neutral = []
    y_neutral = []

    count_positive = 0
    count_negative = 0
    count_neutral = 0

    print("feature engineering...")
    for entry in data:
        if count_positive + count_negative + count_neutral >= number_of_examples:
            break

        text = entry['text']
        label = entry['label']
        preprocessed_text = preprocess_surveys(text)

        if label == 'positive' and count_positive < (number_of_examples / 3):
            X_positive.append(get_context_embedding(preprocessed_text))
            y_positive.append(label)
            count_positive += 1
        elif label == 'negative' and count_negative < (number_of_examples / 3):
            X_negative.append(get_context_embedding(preprocessed_text))
            y_negative.append(label)
            count_negative += 1
        elif label == 'neutral' and count_neutral < (number_of_examples / 3):
            X_neutral.append(get_context_embedding(preprocessed_text))
            y_neutral.append(label)
            count_neutral += 1

    print("Number of positive reviews:", len(X_positive))
    print("Number of negative reviews:", len(X_negative))
    print("Number of neutral reviews:", len(X_neutral))

    return X_positive, y_positive, X_negative, y_negative, X_neutral, y_neutral


def get_tweets_data_for_baselines(number_of_examples):
    df = pd.read_csv('tweets.csv')
    df = df.sample(frac=1).reset_index(drop=True) #shuffle

    X_positive = []
    y_positive = []
    X_negative = []
    y_negative = []
    X_neutral = []
    y_neutral = []

    count_positive = 0
    count_negative = 0
    count_neutral = 0

    print("Feature engineering...")
    for _, row in df.iterrows():
        if count_positive + count_negative + count_neutral >= number_of_examples:
            break

        text = row['Text']
        label = row['sentiment']
        preprocessed_text = preprocess_tweets(text)

        if label == 'positive' and count_positive < (number_of_examples / 3):
            X_positive.append(get_context_embedding(preprocessed_text))
            y_positive.append(label)
            count_positive += 1
        elif label == 'negative' and count_negative < (number_of_examples / 3):
            X_negative.append(get_context_embedding(preprocessed_text))
            y_negative.append(label)
            count_negative += 1
        elif label == 'neutral' and count_neutral < (number_of_examples / 3):
            X_neutral.append(get_context_embedding(preprocessed_text))
            y_neutral.append(label)
            count_neutral += 1

    print("Number of positive reviews:", len(X_positive))
    print("Number of negative reviews:", len(X_negative))
    print("Number of neutral reviews:", len(X_neutral))

    return X_positive, y_positive, X_negative, y_negative, X_neutral, y_neutral
