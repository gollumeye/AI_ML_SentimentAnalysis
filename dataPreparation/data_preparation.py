import json
import re
import random

from transformers import BertTokenizer
from dataPreparation.contractions import contractions
from dataPreparation.feature_engineering import get_context_embedding

def preprocess_text(text_example):
    text_example = text_example.lower()

    # handle contractions like I'll etc.
    words = text_example.split()
    text_example = [contractions[word] if word in contractions else word for word in words]
    text_example = ' '.join(text_example)

    text_example = re.sub(r'[^\w\s]', '', text_example)  # remove punctuation

    return text_example

def get_data_for_bert(num_texts_per_label):
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

    preprocessed_texts = [preprocess_text(text) for text in texts]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = tokenizer(preprocessed_texts, padding=True, truncation=True, return_tensors='pt') #pt -> return pytorch tensor
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2} #bert needs numerical labels
    numerical_labels = [label_map[label] for label in labels]

    return tokenized_texts, numerical_labels

def get_data(number_of_examples):
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
        preprocessed_text = preprocess_text(text)

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