from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob
import numpy as np

class SentimentExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expecting X to be a list or series of strings
        sentiments = []
        for text in X:
            blob = TextBlob(str(text))
            # We extract polarity (-1 to 1) and subjectivity (0 to 1)
            sentiments.append([blob.sentiment.polarity, blob.sentiment.subjectivity])
        return np.array(sentiments)
