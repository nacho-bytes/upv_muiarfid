from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# from gensim.models import KeyedVectors
# w2v_embeddings = KeyedVectors.load_word2vec_format("path-to-embeddings")


def process_text(text, approach, nlp):
    # Approach 1: "basic"
    if approach == "basic":
        text = text.lower()
        return text

    # Approach 2: "spacy"
    if approach == "spacy":
        # Convert the text into a SpaCy Doc object.
        doc = nlp(text)
        # TODO Exercise 1: Preprocess the text using SpaCy.
        # Instructions:
        # 1. Iterate over the tokens in the Spacy Doc object and keep only
        # those tokens that satisfy the following conditions:
        #    * Tokens that contain only alphabetic characters.
        #    * Tokens that are not punctuation.
        #    * Tokens that are not stop-words.
        # 2. For each token in the Doc object, return its lemma (a string),
        # lower-cased, as long as its length is greater or equal than
        # three characters.
        # 3. Return the processed document as a string.
        #

        return text

    # Approach 3: "spacy_pos"
    if approach == "spacy_pos":
        # Convert the text into a SpaCy Doc object.
        doc = nlp(text)
        # TODO Exercise 2: Preprocess the text using SpaCy. Follow the
        # instructions of Exercise 1, with two differences:
        # (1) You should return only tokens that belong to the following POS
        # classes: "NOUN", "VERB", "ADJ", "ADV", "PROPN";
        # (2) Instead of the lemma, return the plain text of the token (`.text`).
        # Again, make sure you return the processed document as a string.
        #

        return text


def train_and_apply_classifier(train_df, curr_test_df, approach):

    # Approach 1:
    if approach == "clf1":
        # Use CountVectorizer to vectorize the data (i.e. to convert your
        # documents to a matrix of token counts). Use the default values.
        vectorizer = CountVectorizer()
        # Learn the vocabulary and vectorize the training set:
        X_train = vectorizer.fit_transform(train_df["processed_text"])
        # Get the train labels:
        y_train = train_df["label"]
        # Vectorize the test set (we don't want to learn the vocabulary
        # mapping again, we will use the vocabulary learned on the training set):
        X_test = vectorizer.transform(curr_test_df["processed_text"])
        # Create a MultinomialNB object (Naive Bayes classifier for multinomial
        # models):
        clf = MultinomialNB()
        # Fit the model to the training data (text + label):
        clf.fit(X_train, y_train)
        # Return predictions for the test set:
        predicted = clf.predict(X_test)
        return predicted

    # Approach 2:
    if approach == "clf2":
        # TODO Exercise 3: Change the classification approach, based on "clf1",
        # following these instructions:
        # * Use TfidfVectorizer to vectorize the data (i.e. to convert your
        #   documents to a matrix of token counts). Use the default values
        #   for the parameters, with the following exceptions:
        #   * Ignore terms that occur in more than 90% of the documents in the dataset.
        #   * Ignore terms that occur in less than 3 documents.
        #   * Extract both unigrams and bigrams.
        # * Use LogisticRegression instead of MultinomialNB.
        #

        return predicted

    # Approach 3:
    if approach == "w2v":
        # TODO Exercise 4 [Optional]: Averaging word embeddings to represent
        # documents as vectors. In order to get a representation of the document
        # using word embeddings, a common (naive) approach is to average the
        # word embeddings of the tokens that occur in the document. In this
        # way, all documents will be represented as a vector of the same size.
        # In order to do this:
        # 1. Download word2vec embeddings in text format. There are many options,
        #    for example: https://fasttext.cc/docs/en/crawl-vectors.html#models
        # 2. Uncomment the two commented lines at the top of this script (the
        #    `KeyedVectors` import on the top of this script and the line below)
        #    and change the path to the embeddings file.
        # 2. For each document in your dataset, tokenize it (split by white
        #    spaces) and get the embedding of each token, if it exists. Then,
        #    average all the token embeddings, so that you end up having just
        #    one embedding per document. All documents should be represented by
        #    an embedding that has the same size.
        # 3. Fit a model (with LogisticRegression, for example) on the training
        #    set, and apply the model to the test set.
        #

        return predicted
