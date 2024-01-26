# Name Detector for Tajik Names

## Overview
The Name Detector is a practical tool designed to identify Tajik full names in text. This functionality is increasingly important in the digital world where personal information is often shared. The tool is versatile, able to detect names not only in the Tajik Cyrillic script but also in the Latin script. This makes it useful in various contexts, such as processing data from social media or official documents.

Its design allows it to recognize names even when they're not in standard formats – it can handle names in lowercase, uppercase, or with typographical errors. This level of flexibility ensures the tool is effective in different scenarios, making it a valuable asset for tasks involving data privacy, such as removing personal information from text before sharing or analyzing it.


## Model Description
The core of the Name Detector project is a sophisticated classifier model specifically designed to recognize Tajik names within text. Below is a detailed description of its key aspects:

### Model Type
- **Model:** CatBoost Classifier
- **Model Framework:** CatBoost, an efficient implementation of gradient boosting.

### Features Used
1. **Name-Based Features:**
   - Counts of how many times substrings of a word match with a predefined list of common Tajik base names.
   - Presence of certain Tajik-specific Cyrillic characters.
   - Normalization of Tajik-specific Cyrillic letters to their Russian equivalents.

2. **Character-Based Features:**
   - Character n-grams ranging from 2 to 6 characters to capture the common character patterns in Tajik names.

3. **Text-Based Features:**
   - Presence of uppercase or title case, which are indicative of names in texts.

### Training Data
- The model is trained on a curated dataset comprising examples of Tajik full names extracted from various sources such as chats, CRM systems, and publicly available name databases.
- The training set includes both positive samples (actual Tajik names) and negative samples (regular text without names).
- The dataset undergoes preprocessing which includes normalization, tokenization, and augmentation to enhance the model's robustness.

### Training Process
- The model is trained using a combination of positive and negative examples to distinguish names from regular text.
- It employs techniques such as oversampling to address class imbalance issues common in such datasets.
- Extensive feature engineering ensures that the model captures the nuances of Tajik names.

## Installation and Usage

### Installation
To install the Name Detector package, follow these steps:

1. **Clone the Repository:**
   Clone the repository to your local machine using Git.
   ```bash
   git clone https://github.com/alifcapital/tajik-name-detector
   ```

2. **Navigate to the Project Directory:**
   Change to the directory where the project is located.
   ```bash
   cd tajik-name-detector
   ```

3. **Install the Package:**
   Install the package using `setup.py`.
   ```bash
   python setup.py install
   ```
   This will install the `name_detector` package along with its dependencies.

### Example Usage
Here's an example script demonstrating how to use the `NameDetector`:

```python
from name_detector import NameDetector

# Initialize the NameDetector
name_detector = NameDetector()

# Text to analyze
text = "Алиҷон Валиев рӯз аз рӯз худро беҳтар ҳис менамуд. Модараш Марям аз ин хушҳол буд."

# Predicts probabilities of being fullname for each sliding window of 2 and 3 words.
windows, probabilities = name_detector.predict(text)

# Display results
for window, probability in zip(windows, probabilities):
    print(f"{window:30s}  {probability:.3f}")
```
