'''
text_processing_utils.py

This script contains the TextProcessor class, which is used for processing text, 
such as lemmatizing and stemming words.
'''

# Standard library imports
import string
import logging
from typing import List, Set

# Third-party imports
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as nltk_stopwords
import spacy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Class for processing text, such as lemmatizing and stemming words.
    """
    def __init__(self):
        """Initializes necessary resources."""
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        self.lemmatizer = spacy.load('en_core_web_sm', disable=["parser", "ner"])
        self.stemmer = SnowballStemmer(language='english')
        self.stopwords = set(nltk_stopwords.words('english'))
        self.punctuation = set(string.punctuation)

    def process_text(self, text: str, method: str) -> Set[str]:
        """
        Processes text by lemmatizing or stemming it and removing stopwords.

        :param text: The text to process.
        :param method: The method to use for processing the text, either 'lemmatize' or 'stem'.
        :return: A set of processed words.
        """
        text.lower()
        processed_words = set()
        if method == 'lemmatize':
            lemmatized_text = self.lemmatizer(text)
            processed_words = {token.lemma_ for token in lemmatized_text if not 
                                (token.is_punct or token.is_space or token.text in self.stopwords)}
        elif method == 'stem':
            words = text.split()
            processed_words = {self.stemmer.stem(word) for word in words if word not in self.stopwords}

        else :
            logger.error('Invalid method for proces_text: %s.'\
                        'Choose between "lemmatize" and "stem".', method)
        return processed_words

    def get_word_variations(self, word: str) -> List[str]:
        """
        Generates variations of a word, including different capitalizations and spaces.
        
        :param word: The word for which to generate variations.
        :return: A list of word variations.
        """
        lemmatized_word = self.lemmatizer(word)[0].lemma_ if len(self.lemmatizer(word)) > 0 else word
        stemmed_word = self.stemmer.stem(word)
        variations = {word, lemmatized_word, stemmed_word}
        variations_capitalization = set()
        for word in variations:
            variations_capitalization.add(word.lower())
            variations_capitalization.add(word.capitalize())
        variations_spaces = set()
        for word in variations_capitalization:
            variations_spaces.add(word)
            variations_spaces.add(' ' + word)
            variations_spaces.add(word + ' ')
            variations_spaces.add(' ' + word + ' ')
        word_variations = [word for word in variations_capitalization if word not in self.stopwords]
        return word_variations
