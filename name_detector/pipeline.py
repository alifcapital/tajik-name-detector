from logging import getLogger
from typing import cast

import tqdm
from joblib import dump, load
from scipy.sparse import hstack

from name_detector.data_preparation import (
    CaseAugmenter,
    LatinAugmenter,
    Preprocessor,
    WordFilter,
    WordSampler,
)
from name_detector.featurizers import CharFeaturizer, NameFeaturizer
from name_detector.utils import (
    count_cyrillic_words,
    create_balanced_train_set,
    load_base_names,
    load_csv_examples,
    load_txt_examples,
)

logger = getLogger()


class TextPipeline:
    # TODO: augment trainset with lowercase examples

    def __init__(self, max_vocab_size: int, base_names: list[str]):
        self.filter = WordFilter()
        self.sampler = WordSampler()
        self.case_augmenter = CaseAugmenter()
        self.latin_augmenter = LatinAugmenter()
        self.preprocessor = Preprocessor()
        self.char_featurizer = CharFeaturizer(max_vocab_size)
        self.name_featurizer = NameFeaturizer(base_names)

        # Intermediate results
        self.filtered_texts: list[str] = []
        self.sampled_texts: list[str] = []
        self.preprocessed_texts: list[list[str]] = []
        self.preprocessed_labels: list = []

    def train(self, data: list[str]):
        self._process_data(data, train=True)
        self.char_featurizer.train(self.preprocessed_texts)

    def transform(self, data: list[str], labels: "list[int]|None" = None, train=False, progress=False):
        self._process_data(data, labels, train=train, progress=progress)
        logger.debug("Featurizing...")
        char_features = self.char_featurizer.transform(self.preprocessed_texts)
        name_features = self.name_featurizer.transform(self.preprocessed_texts)
        labels = self.preprocessed_labels
        features = hstack([char_features, name_features])
        return features, labels

    def get_windows(self, text):
        filtered_text = self.filter.filter(text)

        sampled_texts = self.sampler.sample(filtered_text, None)
        return sampled_texts

    def _process_data(self, data: list[str], labels: "list[int]|None" = None, train=False, progress=False):
        # Reset intermediate results
        self.filtered_texts = []
        self.sampled_texts = []
        self.preprocessed_texts = []
        self.preprocessed_labels = []

        if labels is None:
            labels = [0] * len(data)

        assert len(labels) == len(data)

        iterable = zip(data, labels)
        for text, label in tqdm.tqdm(iterable) if progress else iterable:
            filtered_text = self.filter.filter(text)
            self.filtered_texts.append(filtered_text)

            sampled_texts, sampled_labels = self.sampler.sample(filtered_text, label, sample_one=not train)

            # Data augmentation
            if train:
                sampled_texts, sampled_labels = self.latin_augmenter.augment(sampled_texts, sampled_labels)
                sampled_texts, sampled_labels = self.case_augmenter.augment(sampled_texts, sampled_labels)

            self.sampled_texts.extend(sampled_texts)

            preprocessed_texts = [self.preprocessor.preprocess(text) for text in sampled_texts]
            self.preprocessed_texts.extend(preprocessed_texts)
            self.preprocessed_labels.extend(sampled_labels)

    def save(self, filename: str):
        """
        Save the current state of the TextPipeline object to a file.

        :param filename: The name of the file where the object state will be saved.
        """
        state = {
            "filter": self.filter,
            "sampler": self.sampler,
            "preprocessor": self.preprocessor,
            "char_featurizer": self.char_featurizer,
            "name_featurizer": self.name_featurizer,
        }
        dump(state, filename)

    @classmethod
    def init_from(cls, filename: str):
        """
        Initialize a TextPipeline object from a saved file.

        :param filename: The name of the file to load the object state from.
        :return: An instance of TextPipeline initialized with the saved state.
        """
        state = load(filename)
        # Create a new instance of TextPipeline.
        instance = cls(
            max_vocab_size=state["char_featurizer"].max_vocab_size,
            base_names=state["name_featurizer"].base_names,
        )
        # Restore the state
        instance.filter = state["filter"]
        instance.sampler = state["sampler"]
        instance.preprocessor = state["preprocessor"]
        instance.char_featurizer = state["char_featurizer"]
        instance.name_featurizer = state["name_featurizer"]
        return instance


def prepare_data(config):
    data_dir = config["data_dir"]
    # Load examples
    names_in_chats = load_txt_examples(f"{data_dir}/names_in_chats.txt")
    print(f'Loaded "data/names_in_chats.txt", size: {len(names_in_chats)}')
    crm_names = load_csv_examples(f"{data_dir}/CRM_names.csv")
    print(f'Loaded "data/CRM_names.csv", size: {len(crm_names)}')
    negative_examples = load_csv_examples(f"{data_dir}/chat_messages.csv")
    print(f'Loaded "data/chat_messages.csv", size: {len(negative_examples)}')

    # Filter positive examples: deduplicate
    print("Deduplicating...")
    wfilter = WordFilter()
    names_in_chats = list(set(wfilter.filter(ex) for ex in names_in_chats))
    print(f"names_in_chats: {len(names_in_chats)}")
    crm_names = list(set(wfilter.filter(ex) for ex in crm_names))
    print(f"crm_names: {len(crm_names)}")
    negative_examples = list(set(wfilter.filter(ex) for ex in negative_examples))
    print(f"negative_examples: {len(negative_examples)}")

    # We work with primarily cyrillic texts
    if config["only_cyrillic"]:
        print("Removing non-cyrillic texts...")

        def is_text_mostly_cyrillic(text: str):
            return count_cyrillic_words(text) > len(text.split()) // 2

        names_in_chats = [ex for ex in names_in_chats if is_text_mostly_cyrillic(ex)]
        crm_names = [ex for ex in crm_names if is_text_mostly_cyrillic(ex)]
        negative_examples = [ex for ex in negative_examples if is_text_mostly_cyrillic(ex)]

        print(f"names_in_chats: {len(names_in_chats)}")
        print(f"crm_names: {len(crm_names)}")
        print(f"negative_examples: {len(negative_examples)}")

    positive_test_size = config["chat_names_test_size"]
    crm_samples = config["crm_train_examples"]

    # Split "names in chats" into train and test
    positive_train_examples, positive_test_examples = (
        names_in_chats[positive_test_size:],
        names_in_chats[:positive_test_size],
    )

    # Add train examples from CRM, but balance it with chat_names)
    positive_train_examples = (
        positive_train_examples * (crm_samples // len(positive_train_examples)) + crm_names[:crm_samples]
    )

    # Print final number of positive examples
    print(f"Positive examples. Train: {len(positive_train_examples)}, Test: {len(positive_test_examples)}")

    negative_test_size = config["negative_test_size"]
    negative_train_examples, negative_test_examples = (
        negative_examples[negative_test_size:],
        negative_examples[:negative_test_size],
    )

    return (
        positive_train_examples,
        positive_test_examples,
        negative_train_examples,
        negative_test_examples,
    )


def prepare_pipeline(config):
    (
        positive_train_examples,
        positive_test_examples,
        negative_train_examples,
        negative_test_examples,
    ) = prepare_data(config)

    # Load base names
    base_names = load_base_names(config["data_dir"])
    print(f"Loaded base names, size: {len(base_names)}")

    print("Preparing pipeline...")
    # Create balanced training set for pipeline featurizers
    # TODO still not balanced because it doesnt considering word-tuple sampling
    N = 20000
    pipeline_train_examples, _ = create_balanced_train_set(
        positive_train_examples[: N * 4], negative_train_examples[:N]
    )
    # Create the pipeline
    pipeline = TextPipeline(max_vocab_size=config["vocab_size"], base_names=base_names)

    print("Training pipeline...")
    pipeline.train(pipeline_train_examples)

    pipeline_path = config["pipeline_path"]
    pipeline.save(pipeline_path)
    print(f"Pipeline saved at: {pipeline_path}")

    return {
        "pipeline": pipeline,
        "positive_train_examples": positive_train_examples,
        "positive_test_examples": positive_test_examples,
        "negative_train_examples": negative_train_examples,
        "negative_test_examples": negative_test_examples,
    }


if __name__ == "__main__":
    config = {
        "chat_names_test_size": 3000,
        "crm_train_examples": 10000,
        "vocab_size": 4000,
        "negative_test_size": 10000,
        "pipeline_path": "pipeline.joblib",
        "only_cyrillic": False,
    }
    result = prepare_pipeline(config)

    print("Loading pipeline...")
    pipeline_path: str = cast(str, config["pipeline_path"])
    pipeline = TextPipeline.init_from(pipeline_path)
