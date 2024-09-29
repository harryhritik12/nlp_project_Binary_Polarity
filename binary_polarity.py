import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open("rt-polarity.pos", 'r', encoding='latin-1') as f:
    positive_texts = f.readlines()

with open("rt-polarity.neg", 'r', encoding='latin-1') as f:
    negative_texts = f.readlines()

positive_texts = [text.strip() for text in positive_texts]
negative_texts = [text.strip() for text in negative_texts]

positive_labels = [1] * len(positive_texts)
negative_labels = [0] * len(negative_texts)

texts = positive_texts + negative_texts
labels = positive_labels + negative_labels

data = pd.DataFrame({'text': texts, 'label': labels})

train_pos = positive_texts[:4000]
train_neg = negative_texts[:4000]
val_pos = positive_texts[4000:4500]
val_neg = negative_texts[4000:4500]
test_pos = positive_texts[4500:5331]
test_neg = negative_texts[4500:5331]

train_texts = train_pos + train_neg
train_labels = [1] * len(train_pos) + [0] * len(train_neg)

val_texts = val_pos + val_neg
val_labels = [1] * len(val_pos) + [0] * len(val_neg)

test_texts = test_pos + test_neg
test_labels = [1] * len(test_pos) + [0] * len(test_neg)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

clf = LogisticRegression(max_iter=100000)
clf.fit(X_train, train_labels)

val_preds = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(val_labels, val_preds))
print(classification_report(val_labels, val_preds))

val_conf_matrix = confusion_matrix(val_labels, val_preds)
print("Confusion Matrix (Validation Set):")
print(val_conf_matrix)

test_preds = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(test_labels, test_preds))
print(classification_report(test_labels, test_preds))

test_conf_matrix = confusion_matrix(test_labels, test_preds)
print("Confusion Matrix (Test Set):")
print(test_conf_matrix)

# 0 negative 
# 1 positive 