import numpy as np
import pandas as pd


class TfidfVector:
    def __init__():
        pass

    def _get_term_freq(self, doc):
        """Get the term frequency."""
        tf_dict = {}
        for token in doc:
            if token in tf_dict:
                tf_dict[token] += 1
            else:
                tf_dict[token] = 1
        for token in tf_dict:
            tf_dict[token] = tf_dict[token] / len(doc)
        return tf_dict

    def _get_doc_freq(self, dict):
        """Get the number of documents each token is in."""
        count_dict = {}
        for doc in dict:
            for tok in doc:
                if tok in count_dict:
                    count_dict[tok] += 1
                else:
                    count_dict[tok] = 1
        return count_dict

    def _get_idf(self, dict):
        """Get all unique tokens, along with their idf's."""
        idf_dict = {}
        for tok in dict:
            idf_dict[tok] = np.log(len(dict) / dict[tok])
        return idf_dict

    def fit(self, X, y=None):
        """Get the idf."""
        tf = self._get_term_freq(X)
        doc_freq = self._get_doc_freq(tf)
        self.idf = self._get_idf(doc_freq)
        return self

    def transform(self, X):
        tfidf_dict = {}
        for tok in tfidf_dict:
            tfidf_dict[tok] = tfidf_dict[tok] * self.idf[tok]
        return tfidf_dict
