import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_tfidf(config: dict):
    """
    Combined word + character n-gram TF-IDF.
    Word n-grams capture sentiment phrases.
    Char n-grams capture morphological patterns and handle OOV words.
    """
    cfg = config["classical"]["tfidf"]

    word_tfidf = TfidfVectorizer(
        max_features  = cfg["max_features"],
        ngram_range   = tuple(cfg["ngram_range"]),
        min_df        = cfg["min_df"],
        max_df        = cfg["max_df"],
        sublinear_tf  = cfg["sublinear_tf"],
        analyzer      = "word",
        token_pattern = r"\b[a-zA-Z][a-zA-Z]+\b",
    )

    char_tfidf = TfidfVectorizer(
        max_features  = 20000,
        ngram_range   = (3, 5),
        min_df        = 3,
        max_df        = 0.95,
        sublinear_tf  = True,
        analyzer      = "char_wb",
    )

    return FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf)
    ])


def get_top_features(vectorizer, classifier, label_names, n=20):
    """Get top N predictive features per class."""
    top_features = {}
    offset = 0

    for name, vec in vectorizer.transformer_list:
        feature_names = np.array(vec.get_feature_names_out())
        for i, label in enumerate(label_names):
            coefs   = classifier.coef_[i][offset:offset + len(feature_names)]
            top_idx = np.argsort(coefs)[-n:][::-1]
            if label not in top_features:
                top_features[label] = []
            top_features[label].extend(list(zip(
                feature_names[top_idx],
                coefs[top_idx].round(4)
            )))
        offset += len(feature_names)

    return top_features
