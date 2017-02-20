import argparse
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'


class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="feature options")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument("--holdout", action="store_true",
                        help="Test locally on holdout set")
    parser.add_argument("--holdout_size", type=float, default=0.2,
                        help="Percent of data to use as holdout set")
    parser.add_argument("--no_predict", action="store_false",
                        help="Calculate and save predictions")
    parser.add_argument("--iterations", type=int, default=1,
                        help="How many iterations to do (use with --no_predict)")

    args = parser.parse_args()

    score_sum = 0
    for count in xrange(0, args.iterations):
        # Cast to list to keep it all in memory
        if args.limit > 0:
            raw_train = list(DictReader(open("../data/spoilers/train.csv", 'r')))[:args.limit]
            test = list(DictReader(open("../data/spoilers/test.csv", 'r')))[:args.limit]
        else:
            raw_train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
            test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

        if args.holdout:
            split_point = int(len(raw_train) * args.holdout_size)
            holdout = raw_train[:split_point]
            train = raw_train[split_point:]
        else:
            train = raw_train

        feat = Featurizer()

        labels = []
        for line in train:
            if not line[kTARGET_FIELD] in labels:
                labels.append(line[kTARGET_FIELD])

        print("Label set: %s" % str(labels))
        x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
        x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)
        if args.holdout:
            x_holdout = feat.test_feature(x[kTEXT_FIELD] for x in holdout)
            y_holdout = array(list(labels.index(x[kTARGET_FIELD]) for x in holdout))

        y_train = array(list(labels.index(x[kTARGET_FIELD])
                             for x in train))

        print(len(train), len(y_train))
        print(set(y_train))

        # Train classifier
        lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
        lr.fit(x_train, y_train)

        feat.show_top10(lr, labels)

        if args.holdout:
            score = lr.score(x_holdout, y_holdout)
            score_sum += score
            print(score)

        if args.no_predict:
            print('making predictions')
            predictions = lr.predict(x_test)
            print('saving predictions')
            o = DictWriter(open("predictions.csv", 'w'), ["Id", "spoiler"])
            o.writeheader()
            for ii, pp in zip([x['Id'] for x in test], predictions):
                d = {'Id': ii, 'spoiler': labels[pp]}
                o.writerow(d)

    if args.holdout:
        print("avg score over {} iterations was {}".format(args.iterations, score_sum / args.iterations))
