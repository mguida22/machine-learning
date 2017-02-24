import argparse, string
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline


kTARGET_FIELD = "spoiler"

MURDER_WORDS = ["kill", "kills", "killed",
                "murder", "murdered", "murders",
                "death", "dies", "dead"]
END_WORDS = ["reveal", "revealed", "reveals",
             "turns", "out", "actually", "finale", "end"]

PUNCTUATION = string.punctuation

NAMES = set(line.strip() for line in open('names.txt'))

def remove_punctuation(doc, lower=True, punctuation=PUNCTUATION):
    if lower:
        return "".join([ch for ch in doc if ch not in punctuation]).lower()
    else:
        return "".join([ch for ch in doc if ch not in punctuation])

IMDB = {}
with open("movie_metadata.csv") as csvfile:
    reader = DictReader(csvfile)
    for row in reader:
        key = remove_punctuation(row["movie_title"], lower=False, punctuation=" {}".format(PUNCTUATION))
        IMDB[key] = {
            "duration": row["duration"],
            "gross": row["gross"],
            "genres": row["genres"],
            "plot_keywords": row["plot_keywords"],
            "language": row["language"],
            "country": row["country"],
            "content_rating": row["content_rating"],
            "budget": row["budget"],
            "title_year": row["title_year"],
            "imdb_score": row["imdb_score"]
        }

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return [item[self.key] for item in data]

class KeyWordsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, sentences):
        result = []
        for sentence in sentences:
            end_count = 0
            murder_count = 0
            for word in sentence:
                if any(end_word == word for end_word in END_WORDS):
                    end_count += 1
                if any(murder_word == word for murder_word in MURDER_WORDS):
                    murder_count += 1

            result.append({ "end_word_count": end_count, "murder_count": murder_count })

        return result

class NameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, remove=True, replace_with=""):
        self.remove = remove
        self.replace_with = replace_with

    def fit(self, x, y=None):
        return self

    def transform(self, sentences):
        result = []
        for sentence in sentences:
            sentence = sentence.lower()
            new_sentence = ""
            for word in sentence:
                if not word in NAMES:
                    new_sentence += word
                else:
                    if not self.remove:
                        new_sentence += self.replace_with

                new_sentence += ""

            result.append(sentence)

        return result

class IMDBTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, pages):
        result = []
        for page in pages:
            try:
                obj = IMDB[page]
            except:
                obj = {}

            result.append(obj)

        return result

class PlotKeywordTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, pages):
        result = []
        for page in pages:
            try:
                plot_keywords = IMDB[page]["plot_keywords"]
                plot = [" ".join(word) for word in words.split("|")]
            except:
                plot = "NO_PLOT_NO_PLOT"

            result.append(plot)

        return result

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

    # Need names for analysis, but there has to be a way to programatically get them
    feature_union_names = [
        "sentence",
        # "ngrams_2",
        # "ngrams_3",
        "ngrams_4",
        # "ngrams_5",
        "trope",
        "keywords",
        # "imdb"
    ]

    pipeline = Pipeline([
        ("union", FeatureUnion(
            transformer_list=[
                ("sentence", Pipeline([
                    ("selector", ItemSelector(key="sentence")),
                    ("vectorizer", TfidfVectorizer()),
                ])),

                # ("page", Pipeline([
                #     ("selector", ItemSelector(key="page")),
                #     ("vectorizer", TfidfVectorizer()),
                # ])),
                #
                # ("ngrams_2", Pipeline([
                #     ("selector", ItemSelector(key="sentence")),
                #     ("names", NameTransformer(remove=False, replace_with="NAME_CONSTANT_NAME_CONSTANT")),
                #     ("vectorizer", TfidfVectorizer(preprocessor=remove_punctuation,
                #                                    ngram_range=(2, 2))),
                # ])),
                #
                # ("ngrams_3", Pipeline([
                #     ("selector", ItemSelector(key="sentence")),
                #     ("names", NameTransformer(remove=False, replace_with="NAME_CONSTANT_NAME_CONSTANT")),
                #     ("vectorizer", TfidfVectorizer(preprocessor=remove_punctuation,
                #                                    ngram_range=(3, 3))),
                # ])),

                ("ngrams_4", Pipeline([
                    ("selector", ItemSelector(key="sentence")),
                    ("names", NameTransformer(remove=False, replace_with="NAME_CONSTANT_NAME_CONSTANT")),
                    ("vectorizer", TfidfVectorizer(preprocessor=remove_punctuation,
                                                   ngram_range=(4, 4))),
                ])),
                #
                # ("ngrams_5", Pipeline([
                #     ("selector", ItemSelector(key="sentence")),
                #     ("names", NameTransformer(remove=False, replace_with="NAME_CONSTANT_NAME_CONSTANT")),
                #     ("vectorizer", TfidfVectorizer(preprocessor=remove_punctuation,
                #                                    ngram_range=(5, 5))),
                # ])),

                ("trope", Pipeline([
                    ("selector", ItemSelector(key="trope")),
                    ("vectorizer", CountVectorizer(lowercase=False)),
                ])),

                ("keywords", Pipeline([
                    ("selector", ItemSelector(key="sentence")),
                    ("count_keywords", KeyWordsTransformer()),
                    ("vectorizer", DictVectorizer()),
                ])),

                # ("imdb", Pipeline([
                #     ("selector", ItemSelector(key="page")),
                #     ("imdb", IMDBTransformer()),
                #     ("vectorizer", DictVectorizer())
                # ])),
            ],

            # weight components in FeatureUnion
            transformer_weights = {
                "sentence": 1.0,
                # "page": 1.0,
                # "ngrams_2": 1.0,
                # "ngrams_3": 1.0,
                "ngrams_4": 0.8,
                # "ngrams_5": 1.0,
                "trope": 0.8,
                "keywords": 0.8,
                # "imdb": 1.0,
            },
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
        # Score on holdout set
        score = model.score(holdout, y_holdout)
        print("Accuracy Score:\t{}\n".format(score))

        # Scikit classification_report
        print("Classification Report:\n")
        predictions = model.predict(holdout)
        print(classification_report(predictions, y_holdout, target_names=labels))

        # Confusion Matrix
        print("Confusion Matrix:\n")
        print("\t{}".format("\t".join(labels)))
        cm = confusion_matrix(y_holdout, predictions)
        for i, item in enumerate(cm):
            print("{}\t{}".format(labels[i], "\t".join(str(x) for x in item)))

        # Top/Bottom feature names, for each feature
        classifier_coefs = pipeline.named_steps["classifier"].coef_[0]
        for key in feature_union_names:
            feature = pipeline.named_steps["union"].get_params()[key]
            feature_names = np.asarray(feature.named_steps["vectorizer"].get_feature_names())

            top10 = np.argsort(classifier_coefs[:len(feature_names)])[-10:]
            bottom10 = np.argsort(classifier_coefs[:len(feature_names)])[:10]

            # trim off old coefs as we use them
            classifier_coefs = classifier_coefs[len(feature_names):]

            print("\n--- Predictors for {} ---".format(key))
            print("Pos:\n%s" % ", ".join(feature_names[top10]))
            print("\nNeg:\n%s" % ", ".join(feature_names[bottom10]))
    else:
        print("Predicting and saving test data...")
        predictions = model.predict(test)
        o = DictWriter(open("predictions.csv", 'w'), ["Id", "spoiler"])
        o.writeheader()
        for ii, pp in zip([x['Id'] for x in test], predictions):
            d = {'Id': ii, 'spoiler': labels[pp]}
            o.writerow(d)
