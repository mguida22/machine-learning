import argparse, string
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline


kTARGET_FIELD = "spoiler"

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return [item[self.key] for item in data]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="feature options")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument("--dev", action="store_true",
                        help="Test locally with holdout set, don't actually predict")
    parser.add_argument("--holdout_size", type=float, default=0.2,
                        help="Percent of data to use as holdout set")
    parser.add_argument("--iterations", type=int, default=1,
                        help="How many iterations to do (use with --no_predict)")

    args = parser.parse_args()

    pipeline = Pipeline([
        ("union", FeatureUnion(
            transformer_list=[
                # Pipeline for sentences
                ("sentence", Pipeline([
                    ("selector", ItemSelector(key="sentence")),
                    ("vectorizer", TfidfVectorizer()),
                ])),

                # Pipeline for tropes
                ("trope", Pipeline([
                    ("selector", ItemSelector(key="trope")),
                    ("vectorizer", CountVectorizer(lowercase=False)),
                ])),
            ],
        )),

        ("classifier", SGDClassifier(loss="log", penalty="l2", shuffle=True))
    ])

    train = list(DictReader(open("../data/spoilers/train.csv", "r")))
    test = list(DictReader(open("../data/spoilers/test.csv", "r")))

    if args.limit > 0:
        # we need to get a break point that is close to the limit requested.
        # we can't just use args.limit, because we want the train/holdout sets
        # to be representative of our real test set, where no 'page' present in
        # the train set will be included in the test set
        last_page = ""
        potential_breaks_list = []
        for i, item in enumerate(train):
            if item["page"] != last_page:
                last_page = item["page"]
                potential_breaks_list.append(i)

        index = np.searchsorted(potential_breaks_list, args.limit)
        limit_index = potential_breaks_list[index]

        train = train[:limit_index]
        test = test[:limit_index]

    if args.dev:
        split_point = int(len(train) * args.holdout_size)
        holdout = train[:split_point]
        train = train[split_point:]

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    y_train = array(list(labels.index(item[kTARGET_FIELD]) for item in train))
    if args.dev:
        y_holdout = array(list(labels.index(item[kTARGET_FIELD]) for item in holdout))

    model = pipeline.fit(train, y_train)

    if args.dev:
        score = model.score(holdout, y_holdout)
        print("Score:\t{}".format(score))
        print("Classification Report:\n")
        predictions = model.predict(holdout)
        print(classification_report(predictions, y_holdout, target_names=labels))
    else:
        print("Predicting and saving test data...")
        predictions = model.predict(test)
        o = DictWriter(open("predictions.csv", 'w'), ["Id", "spoiler"])
        o.writeheader()
        for ii, pp in zip([x['Id'] for x in test], predictions):
            d = {'Id': ii, 'spoiler': labels[pp]}
            o.writerow(d)
