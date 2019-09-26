import csv
import operator
import random
import re
import sys
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

"""

# Bag-of-Words Classification with scikit-learn

In this task, you will implement the Bag-of-Words model and text classification in Python.

Implement two methods:
- `preprocess_and_split_to_tokens(sentences) -> tokens_per_sentence`
- `create_bow(sentences, vocab, msg_prefix) -> (vocab, bow_array)`

## Instruction
* See skeleton codes below for more details.
* Do not remove assert lines and do not modify methods that start with an underscore.
* Do not use bag-of-words function implemented in sklearn.
* Before submit your code in KLMS, please change the name of the file to your student id (e.g., 2019xxxx.py).
* Prediction accuracy for unknown test samples (i.e., we do not give them to you) will be your grade.
* Fine-tune your model to get higher accuracy, you can freely choose the pre-processing, model, hyper-parameters, etc.
* See https://scikit-learn.org/stable/modules/classes.html for more information.

"""


def _download_dataset(size=10000):
    assert sys.version_info.major == 3, "Use Python3"

    import ssl
    import urllib.request
    url = "https://raw.githubusercontent.com/dongkwan-kim/small_dataset/master/review_{}k.csv".format(size // 1000)

    dir_path = "../data"
    file_path = os.path.join(dir_path, "review_{}k.csv".format(size // 1000))
    if not os.path.isfile(file_path):
        os.makedirs(dir_path, exist_ok=True)
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as u, open(file_path, 'wb') as f:
            f.write(u.read())
        print("Download: {}".format(file_path))
    else:
        print("Already exist: {}".format(file_path))


def _get_review_data(path, num_samples, train_test_ratio=0.8):
    """Do not modify the code in this function."""
    _download_dataset()
    print("Load Data at {}".format(path))
    reviews, sentiments = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line in reader:
            reviews.append(line["review"])
            sentiments.append(int(line["sentiment"]))

    # Data shuffle
    random.seed(42)
    zipped = list(zip(reviews, sentiments))
    random.shuffle(zipped)
    reviews, sentiments = zip(*(zipped[:num_samples]))
    reviews, sentiments = np.asarray(reviews), np.asarray(sentiments)

    # Train/test split
    num_data, num_train = len(sentiments), int(len(sentiments) * train_test_ratio)
    return (reviews[:num_train], sentiments[:num_train]), (reviews[num_train:], sentiments[num_train:])


def _get_example_of_errors(texts_to_analyze, preds_to_analyze, labels_to_analyze):
    texts_to_analyze = texts_to_analyze[np.random.permutation(len(texts_to_analyze))]
    correct = texts_to_analyze[preds_to_analyze == labels_to_analyze]
    wrong = texts_to_analyze[preds_to_analyze != labels_to_analyze]
    print("\n[Correct Sample Examples]")
    for line in correct[:5]:
        print("\t- {}".format(line))
    print("\n[Wrong Sample Examples]")
    for line in wrong[:5]:
        print("\t- {}".format(line))


def preprocess_and_split_to_tokens(sentences):
    """
    :param sentences: (array_like) array_like objects of strings.
        e.g., ["I like apples", "I love python3"]
    You can choose the level of pre-processing by yourself.
    The easiest way to start is lowering the case (str.lower).

    :return: array_like objects of array_like objects of tokens.
        e.g., [["I", "like", "apples"], ["I", "love", "python3"]]
    """
    stopwords_list = ['i', 'br', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    return [filter(lambda x: x not in stopwords_list, re.sub(r'[^a-z0-9]+', ' ', sentence.lower()).split(' ')) for sentence in sentences]

def create_bow(sentences, vocab=None, msg_prefix="\n"):
    """Make the Bag-of-Words model from the sentences, return (vocab, bow_array)
        vocab: dictionary of (token, index of BoW representation) pair.
        bow_array: array_like objects of BoW representation, the shape of which is [#sentence_list, #vocab]

    :param sentences: (array_like): array_like objects of strings
        e.g., ["I like apples", "I love python3"]
    :param vocab: (dict, optional)
        e.g., {"I": 0, "like": 1, "apples": 2, "love": 3, "python3": 4}
    :param msg_prefix: (str)
    :return: Tuple[dict, array_like]
        e.g., ({"I": 0, "like": 1, "apples": 2, "love": 3, "python3": 4},
                [[1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
    """
    print("{} Bow construction".format(msg_prefix))
    tokens_per_sentence = preprocess_and_split_to_tokens(sentences)
    flattened_list = [y for x in tokens_per_sentence for y in x]
    flattened_set_list = list(set(flattened_list))
    if vocab is None:
        print("{} Vocab construction".format(msg_prefix))
        vocab = {k:v for v,k in enumerate(flattened_set_list)}
    missing_items = flattened_set_list - vocab.keys()
    """
    if missing_items:
        max_key = max(vocab.items(), key=operator.itemgetter(1))[0]
        max_idx = vocab[max_key]
        print("max_idx", max_idx)
        for delta, item in enumerate(missing_items):
            vocab[item] = max_idx + delta + 1
    """
    bow_array = [[0]*len(vocab)] * len(sentences)
    for idx, sentence in enumerate(tokens_per_sentence):
        for word in sentence:
            if vocab.get(word):
                bow_array[idx][vocab[word]] += 1
    return (vocab, bow_array)


def run(test_xs=None, test_ys=None, num_samples=10000, verbose=True):
    """You do not have to consider test_xs and test_ys, since they will be used for grading only."""

    # Data
    (train_xs, train_ys), (val_xs, val_ys) = _get_review_data(path="../data/review_10k.csv", num_samples=num_samples)
    if verbose:
        print("\n[Example of xs]: [\"{}...\", \"{}...\", ...]\n[Example of ys]: [{}, {}, ...]".format(
            train_xs[0][:70], train_xs[1][:70], train_ys[0], train_ys[1]))
        print("\n[Num Train]: {}\n[Num Test]: {}".format(len(train_ys), len(val_ys)))

    # Create bow representation of train set
    my_vocab, train_bows = create_bow(train_xs, msg_prefix="\n[Train]")
    assert isinstance(my_vocab, dict)
    assert isinstance(train_bows, list) or isinstance(train_bows, np.ndarray) or isinstance(train_bows, tuple)
    if verbose:
        print("\n[Vocab]: {} words".format(len(my_vocab)))

    # You can see hyper-parameters (train_kwargs) that can be tuned in the document below.
    #   https://scikit-learn.org/stable/modules/classes.html.
    #pipe = Pipeline([('classifier', RandomForestClassifier())])
    
    #param_grid = [{'classifier': [LogisticRegression()], 'classifier__penalty' : ['l2'], 'classifier__solver' : ['saga']},{'classifier' : [RandomForestClassifier()]}]
    #train_kwargs = dict(verbose=1, penalty='l2', solver="saga") # liblinear, saga
    #clf = LogisticRegression(**train_kwargs)
    #grid_obj = GridSearchCV(pipe, param_grid=param_grid, cv=2, verbose=True)#, n_jobs=-1)
    #grid_obj = grid_obj.fit(train_bows, train_ys)
    #clf = grid_obj.best_estimator_
    #clf = Perceptron(penalty='l2', shuffle=True, verbose=1, class_weight='balanced', alpha=0.0001, eta0=1.0, random_state=0, tol=0.001)
    #clf = MultinomialNB()
    clf = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(2,), tol=0.000000001,verbose=1)
    clf.fit(train_bows, train_ys)
    assert hasattr(clf, "predict")
    # Create bow representation of validation set
    _, val_bows = create_bow(val_xs, vocab=my_vocab, msg_prefix="\n[Validation]")

    # Evaluation
    val_preds = clf.predict(val_bows)
    val_accuracy = accuracy_score(val_ys, val_preds)
    if verbose:
        print("\n[Validation] Accuracy: {}".format(val_accuracy))
        _get_example_of_errors(val_xs, val_preds, val_ys)

    # Grading: Do not modify below lines.
    if test_xs is not None:
        _, test_bows = create_bow(test_xs, vocab=my_vocab, msg_prefix="\n[Test]")
        test_preds = clf.predict(test_bows)
        return {"clf": clf, "val_accuracy": val_accuracy, "test_accuracy": accuracy_score(test_ys, test_preds)}
    else:
        return {"clf": clf}


if __name__ == '__main__':
    run()
