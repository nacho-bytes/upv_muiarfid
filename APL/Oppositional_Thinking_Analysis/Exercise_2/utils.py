import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split


def add_category_columns(df, labels):
    # TODO Exercise 1: Add one column per label to the dataframe (i.e.
    # you need to add 4 columns, called "CAMPAIGNER", "VICTIM", "AGENT",
    # and "FACILITATOR"): its value should be 1 if there is at least one
    # span annotated with this label in the "annotations" field, and 0
    # if there is no span with this label.
    #
    # This function returns a pandas DataFrame.
    
    return


def split_data(df, labels):
    # TODO Exercise 2: Create a train/dev/test split, stratifying by
    # the four labels. You should use 60% for training, 20% for development
    # and 20% for testing. Use 42 as random state.
    #
    # This function returns three pandas DataFrames (one for each split),
    # with the same columns as the input dataframe.

    return


def prepare_data_for_labeling(curr_df, labels, label2id, nlp):
    # TODO Exercise 3: Prepare the data for token classification.
    #
    # The format required to fine-tune a model for token classification
    # using transformers is a dictionary per document, with the following
    # key-value pairs:
    # * "id": the id of the document (int or string).
    # * "tokens": the list of tokens in the document (a list).
    # * "tags": the list of tags associated to the tokens (a list of
    #           the same length as "tokens"). Note that this is neither
    #           the label name (e.g. "AGENT") nor the BIO tag (e.g.
    #           "B-AGENT"): it should be the id of the BIO tag (e.g. 2).
    #
    # In this exercise, the inputs of our classifier will be at the level
    # of the sentence (i.e. not the full document). For that, we need to
    # segment documents into sentences. To do this:
    # 1. Convert each document (column "text") into a SpaCy document.
    # 2. Iterate over the sentences of the Doc object (tip: use `.sents`).
    #    Iterate over the tokens in each sentence, keeping the position
    #    of the token within the document (i.e., token.i) and the token
    #    text (i.e., token.text).
    # 3. For each sentence, return a dictionary with the following fields:
    #    "id", "tokens", and "tags":
    #    * The "id" should be a string with the follwing format: "xxxx_yy",
    #      where `xxxx` is the document id and `yy` is the position (within
    #      the document) of the first token in the sentence.
    #    * The "tokens" value should be a list of the tokens in the sentence.
    #    * The "tags" value should be a list of the BIO tags ids, based on the
    #      values of the "annotations" column in the dataframe (note that we
    #      can map tokens to the annotations because we are using the same
    #      SpaCy model that was used by the organisers to tokenise the dataset).
    #
    # This function returns a list of dictionaries, each dictionary consisting
    # of three keys: "id", "tokens" and "tags".
    #

    return


def apply_model(model_name, test_df, nlp):
    # -------------------------------------------------------------
    # TODO Exercise 5: Given the model name and the test set as a dataframe
    # (and the spaCy object for segmenting documents into sentences), this
    # function returns the output as expected for the shared task evaluation
    # (for more information, see:
    # https://github.com/dkorenci/pan-clef-2024-oppositional/blob/main/README-DATA):
    # "For sequence labeling, the output must be a list of dictionaries,
    # each with 'id', 'annotations' fields. The 'annotations' list can
    # either be empty, or it must contain dictionaries with 'category',
    # 'start_char', 'end_char' fields."
    #
    # Tip: You can use transformers pipelines (task "ner") for that:
    # https://huggingface.co/docs/transformers/main_classes/pipelines.
    # In order to group tokens that belong to the same labels, you can
    # use the "aggregation_strategy" parameter.
    #
    # This function returns the test set formatted as a list of dictionries
    # as required by the shared task.
    
    return
