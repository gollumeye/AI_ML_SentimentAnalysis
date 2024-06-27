import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, matthews_corrcoef, \
    confusion_matrix
from transformers import BertModel
from dataPreparation.data_preparation import get_survey_data_for_bert, get_tweet_data_for_bert
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
import wandb
import pandas as pd
import seaborn as sns


#-----------------------------------------------------------------------------------------------------------------------


TEST_SET_PORTION = 0.2 #must be between 0 and 1
NUMBER_OF_EXAMPLES_PER_LABEL = 1000

TEST_SIZE = (NUMBER_OF_EXAMPLES_PER_LABEL*3)*TEST_SET_PORTION
TRAIN_SIZE = (NUMBER_OF_EXAMPLES_PER_LABEL*3)*(1-TEST_SET_PORTION)

DATASET = 'Tweets' #either 'Tweets' or 'Surveys'

#HYPERPARAMETERS:
LEARNING_RATE = 1e-5
NUMBER_OF_EPOCHS = 5
DROPOUT_RATE = 0.1
BATCH_SIZE = 8

wandb.login(key='dcadd79ea8ec3fd9f6a9ebb81851bcfedd0a1b79')
wandb.init(project='AI_and_ML_project_sentiment_analysis',
           name=f'bert_based_classifier_dataset={DATASET}_num={NUMBER_OF_EXAMPLES_PER_LABEL*3}_lr={LEARNING_RATE}_epochs={NUMBER_OF_EPOCHS}_batchsize={BATCH_SIZE}',
           config={
               "learning_rate": LEARNING_RATE,
               "architecture": "NN",
               "dataset": "Surveys",
               "epochs": NUMBER_OF_EPOCHS,
               "batch_size": BATCH_SIZE,
               "dataset": DATASET
           })


#-----------------------------------------------------------------------------------------------------------------------


"""
Model: BERT Model + additional Linear Layer on top of it
-> Linear Layer is trained for my specific classification task
"""
class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(DROPOUT_RATE) # drop random 10% to prevent overfitting
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
    train_size_per_label = int(train_size/3)
    test_size_per_label = int(test_size/3)
    for label, instances in examples_per_label.items():
        if len(instances) < train_size_per_label + test_size_per_label:
            print(f"Not enough data examples for label {label}")
            print("Number of instances: ", len(instances))
            print("Train + test size: ", train_size_per_label + test_size)
            continue
        train_data.extend(instances[:train_size_per_label])
        test_data.extend(instances[train_size_per_label:train_size_per_label + test_size_per_label])

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

if DATASET == 'Surveys':
    tokenized_texts, numerical_labels = get_survey_data_for_bert(NUMBER_OF_EXAMPLES_PER_LABEL)
else:
    tokenized_texts, numerical_labels = get_tweet_data_for_bert(NUMBER_OF_EXAMPLES_PER_LABEL)

print("split data into test and training set...")
X_train_input_ids, X_train_attention_mask, y_train, X_test_input_ids, X_test_attention_mask, y_test = split_data_into_train_and_test(tokenized_texts, numerical_labels)

model = BERTClassifier(BertModel.from_pretrained('bert-base-uncased'), num_classes=3)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#-----------------------------------------------------------------------------------------------------------------------

"""
Training Loop-
-> One epoch is an iteration where the whole training set is passed through the NN
-> Multiple iterations for updating weights and biases to optimize them
-> In each epoch training data divided into batch
-> Loss calculated for each batch and parameters of model optimized
"""
print("Training BERT Classifier...")
for epoch in range(NUMBER_OF_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUMBER_OF_EPOCHS}")
    total_loss = 0.0
    model.train()
    with tqdm(total=len(X_train_input_ids)) as progress_bar:
        for i in range(0, len(X_train_input_ids), BATCH_SIZE):
            optimizer.zero_grad()
            input_ids = X_train_input_ids[i:i + BATCH_SIZE]
            attention_mask = X_train_attention_mask[i:i + BATCH_SIZE]
            labels = y_train[i:i + BATCH_SIZE]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.update(BATCH_SIZE)
            progress_bar.set_description(f"Training Loss: {loss.item():.4f}")

            wandb.log({'training_loss': loss.item()})

    epoch_loss = total_loss / len(X_train_input_ids)
    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")
    wandb.log({'epoch_loss': epoch_loss, 'epoch': epoch})

#-----------------------------------------------------------------------------------------------------------------------

"""
Testing the Model
"""
print("\nTesting BERT Classifier...")
model.eval()

all_actual_labels = []
all_predicted_labels = []
all_raw_scores = [] #needed for ROC and AUC

with torch.no_grad():
    with tqdm(total=len(X_test_input_ids)) as progress_bar:
        for i in range(0, len(X_test_input_ids), BATCH_SIZE):
            input_ids = X_test_input_ids[i:i + BATCH_SIZE]
            attention_mask = X_test_attention_mask[i:i + BATCH_SIZE]
            raw_scores = model(input_ids=input_ids, attention_mask=attention_mask)

            predicted_labels = torch.argmax(raw_scores, dim=1)
            all_actual_labels.extend(y_test[i:i + BATCH_SIZE].tolist())
            all_predicted_labels.extend(predicted_labels.tolist())
            all_raw_scores.extend(raw_scores.tolist())

            progress_bar.update(BATCH_SIZE)

#-----------------------------------------------------------------------------------------------------------------------

"""
Evaluation
-> Accuracy, Precision, Recall, F1-Score
-> MMC: -1 worst, 0 random, 1 best
-> ROC curve with AUC
"""

print("Class 0: Positive")
print("Class 1: Neutral")
print("Class 2: Negative")

accuracy = accuracy_score(all_actual_labels, all_predicted_labels)
print("Accuracy:", accuracy)
print(classification_report(all_actual_labels, all_predicted_labels))
mcc = matthews_corrcoef(all_actual_labels, all_predicted_labels)
print("MCC:", mcc)


report = classification_report(all_actual_labels, all_predicted_labels, output_dict=True)
metrics = {
    'accuracy': accuracy,
    'mcc': mcc,
    'precision_0': report['0']['precision'],
    'recall_0': report['0']['recall'],
    'f1_score_0': report['0']['f1-score'],
    'precision_1': report['1']['precision'],
    'recall_1': report['1']['recall'],
    'f1_score_1': report['1']['f1-score'],
    'precision_2': report['2']['precision'],
    'recall_2': report['2']['recall'],
    'f1_score_2': report['2']['f1-score'],
}
wandb.log(metrics)

table = wandb.Table(columns=["Metric", "Value"])
table.add_data("Accuracy", accuracy)
table.add_data("MCC", mcc)
for label in [0, 1, 2]:
    table.add_data(f"Precision_{label}", report[str(label)]['precision'])
    table.add_data(f"Recall_{label}", report[str(label)]['recall'])
    table.add_data(f"F1-Score_{label}", report[str(label)]['f1-score'])
wandb.log({"Metrics Table": table})

cm = confusion_matrix(all_actual_labels, all_predicted_labels)
df_cm = pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2])
wandb.log({"confusion_matrix": wandb.Image(sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues"))})

#ROC curve and AUC per class:
y_test_binarized = label_binarize(all_actual_labels, classes=[0, 1, 2])
all_raw_scores = torch.tensor(all_raw_scores).cpu().numpy()
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], all_raw_scores[:, i])
    roc_auc[i] = roc_auc_score(y_test_binarized[:, i], all_raw_scores[:, i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), all_raw_scores.ravel())
roc_auc["micro"] = roc_auc_score(y_test_binarized, all_raw_scores, average="micro")

plt.figure()
colors = cycle(['lightblue', 'orange', 'lightgreen'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC for class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='red', linestyle=':', linewidth=4, label='average ROC curve (AUC = {0:0.2f})'.format(roc_auc["micro"]))

plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.title('ROC')
plt.legend(loc="lower right")

roc_plot_path = "roc_curve.png"
plt.savefig(roc_plot_path)
wandb.log({"roc_curve": wandb.Image(roc_plot_path)})

plt.show()
wandb.finish()