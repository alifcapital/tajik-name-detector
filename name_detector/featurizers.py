import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from name_detector.data_preparation import Preprocessor, WordFilter


class NameFeaturizer:
    PAD_TOKEN = ""
    FEATURE_PER_WORD = 4

    def __init__(self, base_names: list[str]):
        filter = WordFilter()
        self.processor = Preprocessor()
        base_names = [s.lower() for s in base_names]
        base_names = [filter.filter(s) for s in base_names]

        # augment normalized versions
        base_names = base_names + [self.processor.normalize_cyrillic(name) for name in base_names]

        self.base_names = set(base_names)

    def pad_tokens(self, tokens: list[str]):
        if len(tokens) == 3:
            return tokens
        return tokens + [self.PAD_TOKEN] * (3 - len(tokens))

    def featurize_word(self, word: str):
        if word.startswith("_"):
            word = word[1:]
        if word.endswith("_"):
            word = word[:-1]

        word = self.processor.normalize_cyrillic(word)

        is_title = word.istitle()
        isupper = word.isupper()
        word = word.lower()
        base_names = self.base_names
        match_count = 0
        # loop all prefixes
        max_match_size = 0
        for i in range(2, len(word)):
            if word[:i] in base_names:
                match_count += 1
                max_match_size = max(max_match_size, i)

        return [match_count, int(is_title), int(isupper), max_match_size]

    def transform(self, tokenized_texts: list[list[str]]):
        result = []

        for word_list in tokenized_texts:
            features = []
            for word in word_list:
                features.extend(self.featurize_word(word))
            if len(word_list) < 3:
                features += [0, 0, 0, 0] * (3 - len(word_list))
            result.append(features)

        return csr_matrix(result)


class CharFeaturizer:
    PAD_TOKEN = "__"

    def __init__(self, max_vocab_size):
        self.vectorizer_config = dict(
            ngram_range=(2, 6),
            analyzer="char",
            max_features=max_vocab_size,
        )

        self.vectorizer = CountVectorizer(**self.vectorizer_config)

    @property
    def max_vocab_size(self):
        return self.vectorizer_config["max_features"]

    def train(self, data):
        flat_data = [item for sublist in data for item in sublist]
        self.vectorizer.fit(flat_data)
        vocabulary = self.vectorizer.get_feature_names_out()
        if self.PAD_TOKEN not in vocabulary:
            vocabulary = list(vocabulary) + [self.PAD_TOKEN]
            self.vectorizer_config["max_features"] += 1
            self.vectorizer = CountVectorizer(**self.vectorizer_config, vocabulary=vocabulary)

    def pad_tokens(self, tokens: list[str]):
        if len(tokens) == 3:
            return tokens
        return tokens + [self.PAD_TOKEN] * (3 - len(tokens))

    def transform(self, tokenized_texts: list[list[str]]):
        tokenized_texts_flattened = [word for word_list in tokenized_texts for word in self.pad_tokens(word_list)]
        transformed_data = self.vectorizer.transform(tokenized_texts_flattened)
        padded_data = transformed_data
        return np.reshape(padded_data, (padded_data.shape[0] // 3, -1))
