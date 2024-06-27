from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from dataPreparation.data_preparation import get_survey_data_for_baselines
import numpy as np
import wandb
import seaborn as sns
import pandas as pd

NUMBER_OF_EXAMPLES_FOR_BASELINE_MODELS = 30000 #must be dividable by 3

wandb.login(key='dcadd79ea8ec3fd9f6a9ebb81851bcfedd0a1b79')

print("getting data...")
X_positive, y_positive, X_negative, y_negative, X_neutral, y_neutral = get_survey_data_for_baselines(NUMBER_OF_EXAMPLES_FOR_BASELINE_MODELS)

print("splitting data...")
X_train_positive, X_test_positive, y_train_positive, y_test_positive = train_test_split(X_positive, y_positive, test_size=0.2, random_state=0)
X_train_negative, X_test_negative, y_train_negative, y_test_negative = train_test_split(X_negative, y_negative, test_size=0.2, random_state=0)
X_train_neutral, X_test_neutral, y_train_neutral, y_test_neutral = train_test_split(X_neutral, y_neutral, test_size=0.2, random_state=0)

X_train = np.vstack(X_train_positive + X_train_negative + X_train_neutral)
X_test = np.vstack(X_test_positive + X_test_negative + X_test_neutral)
y_train = y_train_positive + y_train_negative + y_train_neutral
y_test = y_test_positive + y_test_negative + y_test_neutral

print("Baseline Models:")

print("------------------------------------")

wandb.init(project='AI_and_ML_project_sentiment_analysis', name=f'rfc_{NUMBER_OF_EXAMPLES_FOR_BASELINE_MODELS}', config={
    "architecture": "RFC",
    "dataset": "Surveys"
    })

print("RANDOM FOREST CLASSIFIER:")
print("training RFC...")
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
print("testing RFC...")
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
wandb.log({"accuracy": accuracy_score(y_test, y_pred), "classification_report": classification_report(y_test, y_pred, output_dict=True)})

wandb.finish()

print("-------------------------------------")

wandb.init(project='AI_and_ML_project_sentiment_analysis', name=f'svm_{NUMBER_OF_EXAMPLES_FOR_BASELINE_MODELS}', config={
    "architecture": "SVM",
    "dataset": "Surveys"
    })

print("SUPPORT VECTOR MACHINE:")
print("training SVM...")
svm = SVC(kernel='linear', random_state=0)
svm.fit(X_train, y_train)
print("testing SVM...")
y_pred = svm.predict(X_test)
print("SVM:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
wandb.log({"accuracy": accuracy_score(y_test, y_pred), "classification_report": classification_report(y_test, y_pred, output_dict=True)})

wandb.finish()

print("-------------------------------------")

wandb.init(project='AI_and_ML_project_sentiment_analysis', name=f'nb_{NUMBER_OF_EXAMPLES_FOR_BASELINE_MODELS}', config={
    "architecture": "Naive Bayes",
    "dataset": "Surveys"
    })

print("NAIVE BAYES:")
nb = GaussianNB()
print("training bayes classifier...")
nb.fit(X_train, y_train)
print("testing bayes classifier...")
y_pred = nb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
wandb.log({"accuracy": accuracy_score(y_test, y_pred), "classification_report": classification_report(y_test, y_pred, output_dict=True)})

wandb.finish()
