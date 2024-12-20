import string

import numpy as np
import pandas as pd
import regex as re
import nltk
from nltk import tokenize as tok
from collections import Counter


# Requires following downloads in parent file
## nltk.download('punkt')
## nltk.download('averaged_perceptron_tagger')
## nltk.download('universal_tagset')

def perform_feature_extraction(df: pd.DataFrame, author_mapping: dict) -> [np.array, np.array]:
    """
    Performs feature, and label extraction on a dataframe consisting of the raw pan data.
    """
    features = np.stack(df["text"].apply(extract_features), dtype=np.float32)
    labels = np.array(list(author_mapping[author] for author in df['author'].values))
    return features, labels


def save_features(features: np.array, labels: np.array, folder_path: str, file_name: str):
    """
    Saves extracted features and labels from perform_feature_extraction to csv files for convenience.
    Saving processing time in re-extracting the features every time the notebook restarts.
    """
    pd.DataFrame(features, columns=None, index=None).to_csv(f'{folder_path}/{file_name}_features', header=False, index=False)
    pd.DataFrame(labels, columns=None, index=None).to_csv(f'./{folder_path}/{file_name}_labels', header=False, index=False)


def load_features(folder_path: str, file_name: str) -> [np.array, np.array]:
    """
    Returns extracted features and labels from csv files saved by save_features.
    """
    features = pd.read_csv(f'{folder_path}/{file_name}_features', header=None).to_numpy(dtype=np.float32)
    labels = pd.read_csv(f'{folder_path}/{file_name}_labels', header=None).to_numpy()
    return features, labels


def extract_features(text: str) -> list:
    """
    Convenience method to extract all features from a text
    :param text: the text to extract features from
    :return: a list of all extracted features
    """
    words = tok.word_tokenize(text)
    sentences = tok.sent_tokenize(text)

    return [
        num_char(text), # 1
        num_sentences(text, sentences), # 2
        num_tokens(text, words), # 3
        num_words_without_vowels(words), # 4
        *special_char_vector(text), # 5-34 (30)
        continuous_punc_count(text), # 35
        contraction_count(text), # 36
        all_caps_words_count(text), # 37
        emoticon_count(text), # 38
        happy_emoticons_count(text), # 39
        sentence_without_capital_start(sentences), # 40
        *pos_tags_proportion(words), # 41-57 (17)
        *letter_frequency(text), # 58-84 (26)
        *function_words_frequency(words), # 85-174 (90)
        small_i_frequency(text) # 175
    ]


def num_char(text: str) -> int:
    """
    Returns the number of characters in the text.
    """
    return len(text)


def num_sentences(text: str, sentences: list[str]) -> int:
    """
    Returns the number of sentences in the text proportional to the number of characters in the text.
    """
    return len(sentences) / num_char(text)


def num_tokens(text: str, words: list[str]) -> int:
    """
    Returns the number of tokens (words) in the text proportional to the number of characters in the text.
    """
    return len(words) / num_char(text)


def num_words_without_vowels(words: list[str]) -> int:
    """
    Returns the number of words in the text that do not contain any vowels. Highly depends on the tokenizer's output.
    """
    word_count = 0
    for word in words:
        if re.search(r'[A-Za-z]', word):
            if not re.search(r'[aeiouAEIOU]', word):
                word_count += 1
    return word_count


def special_char_vector(text: str) -> list[int]:
    """
    Returns a vector of the number of special characters in the text.
    Vector is of (count) shape: 
    [, . / ? : ! % & ( ) digits + - = _ " ' \ @ # $ ` ~ { } [ ] < > |]
    """

    special_chars = [",", ".", "/", "?", ":", "!", "%", "&", "(", ")", "+",
                     "-", "=", "_", '"', "'", "\\", "@", "#", "$", "`", "~", "^",
                     "{", "}", "[", "]", "<", ">", "|"]

    char_count = Counter(text)

    features = [char_count[char] for char in special_chars]
    features.append(sum(char.isdigit() for char in text))

    return features


def continuous_punc_count(text: str) -> int:
    """
    Returns the number of two/three continuous punctuation marks in the text.
    """
    return len(re.findall("[!?.]{2,}", text))


def contraction_count(text: str) -> int:
    """
    Returns the number of contractions in the text.
    """
    return len(re.findall("\w+[\"']\w+", text))


def all_caps_words_count(text) -> int:
    """
    Returns the number of words in the text that are all
    """
    return len(re.findall(r"\b[A-Z]{2,}+\b", text))


def emoticon_count(text: str) -> int:
    """
    Returns the number of emoticons in the text.
    """
    emoticons = [':)', ':(', ':D', ':P', ':o', ':/', '>:(', '^_^', 'T_T', ':-)', ':-(', ':-D', ':-P', ':-o', ':-/',
                 '>:)', '^^', ":'("]
    return sum(text.count(emoticon) for emoticon in emoticons)


def happy_emoticons_count(text: str) -> int:
    """
    Returns the number of happy emoticons in the text.
    """
    happy = [':)', ':D', ':-)', ':-D', '(^_^)', '(^o^)', '(:', '(^.^)', ':]', 'c:', '^^']
    return sum(text.count(emoticon) for emoticon in happy)


def sentence_without_capital_start(sentences: list[str]) -> float:
    """
    Returns the number of sentences that do not start with a capital letter.
    """
    sentences_without_capital_start = [sentence[0].islower() for sentence in sentences]
    return len(sentences_without_capital_start) / len(sentences)


def pos_tags_proportion(words: list[str]) -> list[float]:
    """
    Returns the frequency of different POS tag, proportional to the total number of words.
    """
    pos_tags = nltk.pos_tag(words, tagset='universal')
    pos_tags = [tag for word, tag in pos_tags]

    tags = [
        'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
        'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
    ]

    pos_count = Counter(pos_tags)
    total_length = len(words)

    return [pos_count[tag] / total_length for tag in tags]


def letter_frequency(text: str) -> list[float]:
    """
    Returns the frequency of each letter in the text, proportional to the total number of characters.
    """
    text = text.lower()

    letter_count = Counter(text)
    total_length = len(text)

    return [letter_count[letter] / total_length for letter in string.ascii_lowercase]


def function_words_frequency(words: list[str]) -> list[float]:
    function_words = [
        'a', 'between', 'in', 'nor', 'some', 'upon',
        'about', 'both', 'including', 'nothing', 'somebody', 'us',
        'above', 'but', 'inside', 'of', 'someone', 'used',
        'after', 'by', 'into', 'off', 'something', 'via',
        'all', 'can', 'is', 'on', 'such', 'we',
        'although', 'cos', 'it', 'once', 'than', 'what',
        'am', 'do', 'its', 'one', 'that', 'whatever',
        'among', 'down', 'latter', 'onto', 'the', 'when',
        'an', 'each', 'less', 'opposite', 'their', 'where',
        'and', 'either', 'like', 'or', 'them', 'whether',
        'another', 'enough', 'little', 'our', 'these', 'which',
        'any', 'every', 'lots', 'outside', 'they', 'while',
        'anybody', 'everybody', 'many', 'over', 'this', 'who',
        'anyone', 'everyone', 'me', 'own', 'those', 'whoever',
        'anything', 'everything', 'more', 'past', 'though', 'whom'
    ]

    function_word_count = Counter(words)
    total_length = len(words)

    return [function_word_count[word] / total_length for word in function_words]


def small_i_frequency(text: str) -> int:
    """
    Returns the frequency of each letter in the text, proportional to the total number of characters.
    """
    return len(re.findall("[,. ]i[',. ]", text))
