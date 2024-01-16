import re

from name_detector.utils import is_cyrillic, latinize_text


class WordFilter:
    def __init__(self):
        self.non_word_pattern = re.compile(r"\W+")

    def filter(self, text):
        return re.sub(self.non_word_pattern, " ", text).strip()


class OneToManyAugmenter:
    def _augment(self, text: str, label=None) -> "tuple[list, list] | list":
        raise NotImplementedError

    def augment(self, texts: list[str], labels: "list|None"):
        if labels is not None:
            assert len(labels) == len(texts)

        new_texts: list[str] = []
        new_labels: list = []

        for text, label in zip(texts, labels if labels is not None else [None] * len(texts)):
            augmented_texts, augmented_labels = self._augment(text, label)
            if labels is not None:
                new_labels.extend(augmented_labels)
            new_texts.extend(augmented_texts)

        return new_texts, None if labels is None else new_labels


class CaseAugmenter(OneToManyAugmenter):
    def _augment(self, text: str, label=None):
        augmented_texts = [text]
        augmented_labels = [label]

        if not text.islower():
            augmented_texts.append(text.lower())
            augmented_labels.append(label)
        return augmented_texts, augmented_labels


class LatinAugmenter(OneToManyAugmenter):
    def _augment(self, text: str, label=None):
        augmented_texts = [text]
        augmented_labels = [label]

        if is_cyrillic(text):
            latin_text = latinize_text(text)
            augmented_texts.append(latin_text)
            augmented_labels.append(label)

        return augmented_texts, augmented_labels


class WordSampler:
    def sample(self, text: str, label=None, sample_one=False):
        """
        Samples 2-3 word combinations from the given text and returns them with corresponding labels.

        :param text: The input text string.
        :param label: The label associated with the text.
        :param sample_one: Sample only one. If text is long, then maximum three words are returned
        :return: A tuple of two lists - sampled texts and their corresponding labels.
        """
        words = text.split()
        if len(words) < 2:
            sampled_labels, sampled_texts = [], []
        elif sample_one:
            sampled_texts = [" ".join(words[:3])]
            sampled_labels = [label]
        else:
            # sampled_texts = [' '.join(words[i:i+n]) for n in range(2, 4) for i in range(len(words)-n+1)]
            sampled_texts = [
                " ".join(words[i : i + n]) for i in range(len(words)) for n in range(2, 4) if i + n <= len(words)
            ]
            sampled_labels = [label] * len(sampled_texts)
        return (sampled_texts, sampled_labels) if label is not None else sampled_texts


class Preprocessor:
    def normalize_cyrillic(self, text):
        """
        Normalize Tajik-specific Cyrillic letters to Russian versions.

        :param text: The text to be normalized.
        :return: Normalized text.
        """
        # Mapping of Tajik-specific Cyrillic letters to Russian equivalents
        # This example includes a few letters - you may need to expand this based on specific requirements
        tajik_to_russian = {
            "ӣ": "и",
            "ӯ": "у",
            "Ӯ": "У",
            "ҳ": "х",
            "Ҳ": "Х",
            "қ": "к",
            "Қ": "К",
            "ғ": "г",
            "Ғ": "Г",
            "ҷ": "ч",
            "Ҷ": "Ч",
        }

        # Replace each Tajik-specific letter with its Russian equivalent
        for tajik_letter, russian_letter in tajik_to_russian.items():
            text = text.replace(tajik_letter, russian_letter)

        return text

    def tokenize(self, text: str):
        return text.split()

    def preprocess(self, text: str):
        text = self.normalize_cyrillic(text)
        return self.tokenize(text)
