import spacy
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split

# Load a SpaCy model: https://spacy.io/usage/models
nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])  # Spanish
# nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"]) # English

import utils

# -----------------------------------------------------------------
# 1. Loading the data
# -----------------------------------------------------------------

# Load the json into a pandas dataframe:
df = pd.read_json("dataset_oppositional/dataset_es_train.json")  # Spanish
# df = pd.read_json("dataset_oppositional/dataset_en_train.json") # English

# We add a new column where we map the `category` to a numerical value.
# We will consider `CONSPIRACY` as the positive class:
df["label"] = df["category"].map({"CONSPIRACY": 1, "CRITICAL": 0})

# For the binary classification task, we only need the following columns:
df = df[["id", "text", "label"]]

# -----------------------------------------------------------------
# 2. Preprocessing the data
# -----------------------------------------------------------------

# Apply the preprocessing function to the "text" column of the dataframe,
# specifying the preprocessing approach, and store it in a new column
# called "processed_text". Three different preprocessing approaches
# are allowed: "basic", "spacy", and "spacy_pos": change the value of
# `approach` accordingly. The values of the "text" column are strings
# and the values of the "processed_text" column should also be strings.
df["processed_text"] = df.apply(
    lambda x: utils.process_text(x["text"], approach="basic", nlp=nlp), axis=1
)

# -----------------------------------------------------------------
# 3. Splitting the dataset into training, development and test sets
# -----------------------------------------------------------------

# Two-step splitting into training, development and test. We'll have:
# * 0.60 for training
# * 0.20 for development
# * 0.20 for testing
# Stratifying by label makes sure we have similar class frequencies:
X_train, X_tmp, y_train, y_tmp = train_test_split(
    df["id"], df["label"], stratify=df["label"], test_size=0.4, random_state=42
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_tmp, y_tmp, stratify=y_tmp, test_size=0.5, random_state=42
)
train_df = df[df["id"].isin(X_train)]
dev_df = df[df["id"].isin(X_dev)]
test_df = df[df["id"].isin(X_test)]

# -----------------------------------------------------------------
# 4. Training, applying, and evaluating a classifier (on the dev set)
# -----------------------------------------------------------------

predicted = utils.train_and_apply_classifier(train_df, dev_df, approach="clf1")

mcc = matthews_corrcoef(dev_df["label"], predicted)
p = precision_score(dev_df["label"], predicted, average="macro")
r = recall_score(dev_df["label"], predicted, average="macro")
f1_crit = f1_score(dev_df["label"], predicted, average="binary", pos_label=0)
f1_cons = f1_score(dev_df["label"], predicted, average="binary", pos_label=1)
f1 = f1_score(dev_df["label"], predicted, average="macro")

print("MCC:", round(mcc, 3))
print("Precision:", round(p, 3))
print("Recall:", round(r, 3))
print("F1-score critical:", round(f1_crit, 3))
print("F1-score conspiracy:", round(f1_cons, 3))
print("F1-score macro:", round(f1, 3))
