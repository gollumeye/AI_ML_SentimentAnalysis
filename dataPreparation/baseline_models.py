from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from data_preparation import get_data
import numpy as np

print("getting data...")
X_positive, y_positive, X_negative, y_negative, X_neutral, y_neutral = get_data()

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

print("RANDOM FOREST CLASSIFIER:")
print("training RFC...")
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
print("testing RFC...")
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("-------------------------------------")

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

print("-------------------------------------")

print("NAIVE BAYES:")
nb = GaussianNB()
print("training bayes classifier...")
nb.fit(X_train, y_train)
print("testing bayes classifier...")
y_pred = nb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

"""
print("Choose model:")
print("1: Random Forest Classifier")
print("2: Support Vector Machine")
print("3: Naive Bayes Classifier")
choice = input("Enter number: ")

if choice == '1':
    print("training RFC...")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)

    print("testing RFC...")
    y_pred = clf.predict(X_test)
    print("Random Forest:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


elif choice == '2':

    print("training SVM...")
    svm = SVC(kernel='linear', random_state=0)
    svm.fit(X_train, y_train)

    print("testing SVM...")
    y_pred = svm.predict(X_test)
    print("SVM:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

elif choice == '3':
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    print("Naive Bayes:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
"""
