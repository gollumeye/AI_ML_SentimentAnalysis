import json
import re
from contractions import contractions
from feature_engineering import get_context_embedding

NUMBER_OF_EXAMPLES = 3000 #should be dividable by 3

def preprocess_text(text_example):
    text_example = text_example.lower()

    # handle contractions like I'll etc.
    words = text_example.split()
    text_example = [contractions[word] if word in contractions else word for word in words]
    text_example = ' '.join(text_example)

    text_example = re.sub(r'[^\w\s]', '', text_example)  # remove punctuation

    return text_example


def get_data():
    with open('../surveys.json', 'r') as file:
        data = json.load(file)

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
        if count_positive + count_negative + count_neutral >= NUMBER_OF_EXAMPLES:
            break

        text = entry['text']
        label = entry['label']
        preprocessed_text = preprocess_text(text)

        if label == 'positive' and count_positive < (NUMBER_OF_EXAMPLES/3):
            X_positive.append(get_context_embedding(preprocessed_text))
            y_positive.append(label)
            count_positive += 1
        elif label == 'negative' and count_negative < (NUMBER_OF_EXAMPLES/3):
            X_negative.append(get_context_embedding(preprocessed_text))
            y_negative.append(label)
            count_negative += 1
        elif label == 'neutral' and count_neutral < (NUMBER_OF_EXAMPLES/3):
            X_neutral.append(get_context_embedding(preprocessed_text))
            y_neutral.append(label)
            count_neutral += 1

    print("Number of positive reviews:", len(X_positive))
    print("Number of negative reviews:", len(X_negative))
    print("Number of neutral reviews:", len(X_neutral))

    return X_positive, y_positive, X_negative, y_negative, X_neutral, y_neutral