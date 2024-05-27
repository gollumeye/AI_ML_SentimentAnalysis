import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertModel
from data_preparation import get_data_for_bert
from tqdm import tqdm
import random

TRAIN_SIZE = 40
TEST_SIZE = 10


"""
Model: BERT Model + additional Linear Layer on top of it
-> Linear Layer is trained for my specific classification task
"""
class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        # self.dropout = torch.nn.Dropout(0.1) # drop random 10% to prevent overfitting
        self.sentimentClassifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):  # attention mask specifies which of the tokens from input_ids should be taken into account
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # get output from BERT model
        pooled_output = outputs.pooler_output
        raw_scores = self.sentimentClassifier(pooled_output)  # pass output from BERT to classifier to get line vulnerability predictions
        return raw_scores


#-----------------------------------------------------------------------------------------------------------------------


"""
Data Preparation
-> Loading data from dataset
-> Preprocessing texts
-> Splitting into test and (balanced) training set
"""
def split_data_into_train_and_test(tokenized_texts, numerical_labels, train_size=TRAIN_SIZE, test_size=TEST_SIZE):
    input_ids = tokenized_texts['input_ids']
    attention_mask = tokenized_texts['attention_mask']

    all_examples = [(input_ids[i], attention_mask[i], numerical_labels[i]) for i in range(len(input_ids))]
    random.shuffle(all_examples)

    examples_per_label = {0: [], 1: [], 2: []}
    for input_ids, attention_mask, label in all_examples:
        examples_per_label[label].append((input_ids, attention_mask, label))

    train_data = []
    test_data = []
    for label, instances in examples_per_label.items():
        if len(instances) < train_size + test_size:
            print(f"Not enough data examples for label {label}")
            continue
        train_data.extend(instances[:train_size])
        test_data.extend(instances[train_size:train_size + test_size])

    if not train_data or not test_data:
        raise ValueError("Not enough data for training or testing.")

    random.shuffle(train_data)
    random.shuffle(test_data)

    X_train_input_ids = torch.stack([entry[0] for entry in train_data])
    X_train_attention_mask = torch.stack([entry[1] for entry in train_data])
    y_train = torch.tensor([entry[2] for entry in train_data])
    X_test_input_ids = torch.stack([entry[0] for entry in test_data])
    X_test_attention_mask = torch.stack([entry[1] for entry in test_data])
    y_test = torch.tensor([entry[2] for entry in test_data])

    return X_train_input_ids, X_train_attention_mask, y_train, X_test_input_ids, X_test_attention_mask, y_test


print("get data...")
tokenized_texts, numerical_labels = get_data_for_bert(num_texts_per_label=50)

print("split data into test and training set...")
X_train_input_ids, X_train_attention_mask, y_train, X_test_input_ids, X_test_attention_mask, y_test = split_data_into_train_and_test(tokenized_texts, numerical_labels)

model = BERTClassifier(BertModel.from_pretrained('bert-base-uncased'), num_classes=3)
optimizer = optim.Adam(model.parameters(), lr=1e-5)


#-----------------------------------------------------------------------------------------------------------------------


"""
Training Loop-
-> One epoch is an iteration where the whole training set is passed through the NN
-> Multiple iterations for updating weights and biases to optimize them
-> In each epoch training data divided into batch
-> Loss calculated for each batch and parameters of model optimized
"""
print("Training BERT Classifier...")
num_epochs = 30
batch_size = 8
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_loss = 0.0
    model.train()
    with tqdm(total=len(X_train_input_ids)) as progress_bar:
        for i in range(0, len(X_train_input_ids), batch_size):
            optimizer.zero_grad()
            input_ids = X_train_input_ids[i:i + batch_size]
            attention_mask = X_train_attention_mask[i:i + batch_size]
            labels = y_train[i:i + batch_size]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.update(batch_size)
            progress_bar.set_description(f"Training Loss: {loss.item():.4f}")

    epoch_loss = total_loss / len(X_train_input_ids)
    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")


#-----------------------------------------------------------------------------------------------------------------------


"""
Testing the Model
"""
print("\nTesting BERT Classifier...")
model.eval()

all_actual_labels = []
all_predicted_labels = []

with torch.no_grad():
    with tqdm(total=len(X_test_input_ids)) as progress_bar:
        for i in range(0, len(X_test_input_ids), batch_size):
            input_ids = X_test_input_ids[i:i + batch_size]
            attention_mask = X_test_attention_mask[i:i + batch_size]
            raw_scores = model(input_ids=input_ids, attention_mask=attention_mask)

            predicted_lables = torch.argmax(raw_scores, dim=1)
            all_actual_labels.extend(y_test[i:i + batch_size].tolist())
            all_predicted_labels.extend(predicted_lables.tolist())

            progress_bar.update(batch_size)


#-----------------------------------------------------------------------------------------------------------------------


"""
Evaluation
-> Accuracy, Precision, Recall, F1-Score
"""
print("Accuracy:", accuracy_score(all_actual_labels, all_predicted_labels))
print(classification_report(all_actual_labels, all_predicted_labels))
