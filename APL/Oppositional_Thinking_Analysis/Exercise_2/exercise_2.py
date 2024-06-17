import utils

import numpy as np
import pandas as pd

from datasets import Dataset, load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

import spacy

nlp = spacy.load("en_core_web_sm")


# -------------------------------------------------------------
# Load the data, keep only "CONSPIRACY" documents:

df = pd.read_json("dataset_oppositional/dataset_en_train.json")
df = df[df["category"] == "CONSPIRACY"].reset_index()
df = df[["id", "text", "annotations"]]

print(df.shape == (1379, 3))


# -------------------------------------------------------------
# Define the labels, the BIO tags, and the id2label and label2id dicts:

labels = ["CAMPAIGNER", "VICTIM", "AGENT", "FACILITATOR"]
bio_tags = ["O"] + ["B-" + x for x in labels] + ["I-" + x for x in labels]

id2label = dict()
for i in range(len(bio_tags)):
    id2label[i] = bio_tags[i]
label2id = {v: k for k, v in id2label.items()}

print(label2id["O"] == 0)
print(label2id["I-FACILITATOR"] == 8)


# -------------------------------------------------------------
# Add a column for each category:
# [TODO in utils.py (exercise 1)]

filt_df = utils.add_category_columns(df, labels)

print(filt_df.shape == (1379, 7))

print(filt_df.iloc[5].id == 6747)
print(filt_df.iloc[5].CAMPAIGNER == 1)
print(filt_df.iloc[5].VICTIM == 0)
print(filt_df.iloc[5].AGENT == 1)
print(filt_df.iloc[5].FACILITATOR == 0)


# -------------------------------------------------------------
# Create a training, development and test split:
# [TODO in utils.py (exercise 2)]

train_df, dev_df, test_df = utils.split_data(filt_df, labels)

print(train_df.shape == (827, 7))
print(dev_df.shape == (276, 7))
print(test_df.shape == (276, 7))


# -------------------------------------------------------------
# Prepare the data for token classification:
# [TODO in utils.py (exercise 3)]

train_seq_data = utils.prepare_data_for_labeling(train_df, labels, label2id, nlp)
dev_seq_data = utils.prepare_data_for_labeling(dev_df, labels, label2id, nlp)
test_seq_data = utils.prepare_data_for_labeling(test_df, labels, label2id, nlp)

print(len(train_seq_data) == 7621)
print(len(dev_seq_data) == 2759)
print(len(test_seq_data) == 2438)
print(train_seq_data[110]["id"] == "10723_25")
print(train_seq_data[110]["tokens"] == ["This",
                                        "man",
                                        "can",
                                        "claim",
                                        "up",
                                        "to",
                                        "£",
                                        "120,000",
                                        "for",
                                        "medical",
                                        "battery",
                                        ".",])
print(train_seq_data[110]["tags"] == [2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


# -------------------------------------------------------------
# Convert the data to Dataset format:

hf_train = Dataset.from_list(train_seq_data)
hf_dev = Dataset.from_list(dev_seq_data)
hf_test = Dataset.from_list(test_seq_data)


# -------------------------------------------------------------
# Fine-tune DistilBERT for token classification:
# [TODO Exercise 4]
#
# To fine-tune DistilBERT for token classification, you can follow the
# steps in the following notebook tutorial by HuggingFace (note that the
# data loading is already done, you can mostly skip until the "Preprocessing
# the data" section):
# * https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb
# (License: https://www.apache.org/licenses/LICENSE-2.0.html)
#
# You can find more information in the following docs:
# * https://huggingface.co/learn/nlp-course/chapter7/2
# * https://huggingface.co/docs/transformers/tasks/token_classification
#
# Instructions:
# * You don't need to log in to HuggingFace or push the model to the hub.
#   You can save it locally instead.
# * You won't submit the code for this exercise. You only need to provide
#   a small report describing the approach, the results on the test set
#   (report the F1-score, precision and recall both for overall performance
#   and for each category) and a couple of suggestions for improvig the
#   performance.
# * You should use "distilbert-base-uncased" as base model, train it on
#   three epochs, and the resulting fine-tuned model should be called
#   distilbert-finetuned-oppo.




# -------------------------------------------------------------
# Apply model to the test set:
# [TODO in utils.py (exercise 5)]

model_name = "distilbert-finetuned-oppo"
test_results = utils.apply_model(model_name, test_df, nlp)
print(test_results[0])
