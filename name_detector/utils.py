import re
import unicodedata

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


def load_csv_examples(csv_path):
    """
    Load negative examples from a CSV file.

    :param csv_path: Path to the CSV file.
    :return: List of negative examples.
    """
    try:
        df = pd.read_csv(csv_path, header=None)[0]
        df = df.dropna()
        df = df.astype(str)
        return df.tolist()
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return []
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return []


def load_txt_examples(txt_path):
    """
    Load positive examples from a TXT file.

    :param txt_path: Path to the TXT file.
    :return: List of positive examples.
    """
    try:
        with open(txt_path, "r") as file:
            pos_examples = file.readlines()
        pos_examples = [line.strip() for line in pos_examples]
        return pos_examples
    except FileNotFoundError:
        print(f"File not found: {txt_path}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []


def load_base_names(data_dir: str):
    with open(f"{data_dir}/base_names.txt") as f:
        result = f.read().splitlines()

    def process_name(s):
        s = re.sub(r"\W+", "", s).strip()
        s = s.title()
        return s

    df = pd.read_csv(f"{data_dir}/names_data.csv")
    other_names = (
        df["Tajik"].apply(process_name).tolist()
        + df["Russian"].apply(process_name).tolist()
        + df["English"].apply(process_name).tolist()
    )

    return list(set(result + other_names))


class Transliterator:
    # Create mapping
    translit_dict = {
        "yo": "ё",
        "yu": "ю",
        "ya": "я",
        "sh": "ш",
        "ch": "ч",
        "gh": "ғ",
        "zh": "ж",
        "q": "қ",
        "j": "ҷ",
        "h": "ҳ",
        "a": "а",
        "b": "б",
        "v": "в",
        "g": "г",
        "d": "д",
        "e": "е",
        "z": "з",
        "i": "и",
        "y": "й",
        "k": "к",
        "l": "л",
        "m": "м",
        "n": "н",
        "o": "о",
        "p": "п",
        "r": "р",
        "s": "с",
        "t": "т",
        "u": "у",
        "f": "ф",
        "c": "к",
        "y": "й",
        "e": "е",
    }

    # Special cases must be replaced before single characters
    special_cases = [k for k in translit_dict.keys() if len(k) > 1]

    @classmethod
    def transliterate(cls, text: str):
        translit_dict = cls.translit_dict
        text = text.lower()
        for special in cls.special_cases:
            if special in text:
                text = text.replace(special, translit_dict[special])

        transliterated_text = ""
        for char in text:
            if char in translit_dict:
                transliterated_text += translit_dict[char]
            else:
                transliterated_text += char
        return transliterated_text


def latinize_text(name: str) -> str:
    # Define the Cyrillic-to-Latin character mappings
    cyrillic_to_latin_map = {
        "А": "A",
        "а": "a",
        "Б": "B",
        "б": "b",
        "В": "V",
        "в": "v",
        "Г": "G",
        "г": "g",
        "Д": "D",
        "д": "d",
        "Е": "E",
        "е": "e",
        "Ё": "Yo",
        "ё": "yo",
        "Ж": "Zh",
        "ж": "zh",
        "З": "Z",
        "з": "z",
        "И": "I",
        "и": "i",
        "Й": "Y",
        "й": "y",
        "К": "K",
        "к": "k",
        "Л": "L",
        "л": "l",
        "М": "M",
        "м": "m",
        "Н": "N",
        "н": "n",
        "О": "O",
        "о": "o",
        "П": "P",
        "п": "p",
        "Р": "R",
        "р": "r",
        "С": "S",
        "с": "s",
        "Т": "T",
        "т": "t",
        "У": "U",
        "у": "u",
        "Ф": "F",
        "ф": "f",
        "Х": "Kh",
        "х": "kh",
        "Ц": "Ts",
        "ц": "ts",
        "Ч": "Ch",
        "ч": "ch",
        "Ш": "Sh",
        "ш": "sh",
        "Щ": "Shch",
        "щ": "shch",
        "Ъ": "",
        "ъ": "",
        "Ы": "Y",
        "ы": "y",
        "Ь": "",
        "ь": "",
        "Э": "E",
        "э": "e",
        "Ю": "Yu",
        "ю": "yu",
        "Я": "Ya",
        "я": "ya",
        # Tajik-specific characters
        "Ҷ": "J",
        "ҷ": "j",
        "Ғ": "Gh",
        "ғ": "gh",
        "Қ": "Q",
        "қ": "q",
        "Ҳ": "H",
        "ҳ": "h",
        "Ӯ": "U",
        "ӯ": "u",
        "Ӣ": "I",
        "ӣ": "i",
    }

    # Replace each Cyrillic character in the input string with its Latin equivalent
    latinized_text = ""
    for char in name:
        latinized_text += cyrillic_to_latin_map.get(
            char, char
        )  # Default to the original character if not found in the map

    return latinized_text


def is_cyrillic(s: str):
    """Check if all words in a given string are written in Cyrillic script."""
    s = "".join(char for char in s if char.isalpha())
    return all("CYRILLIC" in unicodedata.name(ch) for ch in s)


def count_cyrillic_words(s):
    """
    Counts how many words in the input string are composed entirely of Cyrillic characters.

    :param s: A string containing words.
    :return: The count of words composed entirely of Cyrillic characters.
    """
    # Split the string into words
    s = re.sub(r"\W+", " ", s).strip()  # remove non-words
    words = s.split()

    # Count how many words are Cyrillic
    return sum(is_cyrillic(word) for word in words)


def train_test_split(examples, test_size=20000):
    """
    Splits the examples into training and test sets.

    :param examples: List of examples.
    :param test_size: Number of examples to include in the test set for examples.
    :return: A tuple of two lists: (train, test).
    """
    # Ensure there are enough examples for the specified test sizes
    if len(examples) < test_size:
        raise ValueError("Not enough examples to meet the specified test sizes")

    # Split examples
    train = examples[:-test_size]
    test = examples[-test_size:]

    return train, test


def create_balanced_train_set(pos_train, neg_train):
    # Combine positive and negative training examples
    X_train = pos_train + neg_train
    y_train = [1] * len(pos_train) + [0] * len(neg_train)  # 1 for positive, 0 for negative

    # Oversample the negative examples
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(np.array(X_train).reshape(-1, 1), y_train)

    # Flatten the resampled data
    X_train_resampled = X_train_resampled.flatten().tolist()
    return X_train_resampled, y_train_resampled


if __name__ == "__main__":
    tr = Transliterator.transliterate("Pochemu transakciya cherez prilozhenie uzhe 2 ")
    # tr = transliterate('салом! korti manba pul gzaron')
    print(tr)
