from math import pi, sin

kSIMPLE_TRAIN = [(1, False), (2, True), (4, False), (5, True), (13, False),
                 (14, True), (19, False)]


class SinClassifier:
    """
    A binary classifier that is parameterized a single float
    """

    def __init__(self, w):
        """
        Create a new classifier parameterized by w

        Args:
          w: The parameter w in the sin function (a real number)
        """
        assert isinstance(w, float)
        self.w = w

    def __call__(self, k):
        """
        Returns the raw output of the classifier.  The sign of this value is the
        final prediction.

        Args:
          k: The exponent in x = 2**(-k) (an integer)
        """
        return sin(self.w * 10 ** (-k))

    def classify(self, k):
        """

        Classifies an integer exponent based on whether the sign of \sin(w * 2^{-k})
        is >= 0.  If it is, the classifier returns True.  Otherwise, false.

        Args:
          k: The exponent in x = 2**(-k) (an integer)
        """
        assert isinstance(k, int), "Object to be classified must be an integer"

        if self(k) >= 0:
            return True
        else:
            return False


def train_sin_classifier(data):
    """
    Compute the correct parameter w of a classifier to prefectly classify the
    data and return the corresponding classifier object

    Args:
      data: A list of tuples; first coordinate is k (integers), second is y (+1/-1)
    """

    assert all(isinstance(k[0], int) and k >= 0 for k in data), \
        "All training inputs must be integers"
    assert all(isinstance(k[1], bool) for k in data), \
        "All labels must be True / False"

    # TODO: Compute a parameter w that will correctly classify the dataset

    # Following from the reading....
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/svmtutorial.pdf
    suml = 0.0
    for item in data:
        suml += (1 - item[1]) * (10 ** item[0])

    w = pi * (1 + suml)
    return SinClassifier(w)

if __name__ == "__main__":
    classifier = train_sin_classifier(kSIMPLE_TRAIN)
    for kk, yy in kSIMPLE_TRAIN:
        print(kk, yy, classifier(kk), classifier.classify(kk))
