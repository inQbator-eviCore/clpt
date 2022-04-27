"""NLP DocumentCleaner stage for cleaning the CLAO.

DocumentCleaner includes removing stop words, converting to lower case, excluding punctuations and checking spelling.
"""
import re
from abc import abstractmethod

from overrides import overrides
# from pattern.en import lemma
from sklearn.feature_extraction import text

from src.clao.clao import TextCLAO
from src.constants.annotation_constants import RAW_TEXT
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage

STOPWORD = '<stopword>'


class DocumentCleaner(PipelineStage):
    """Clean the raw texts in CLAO."""
    @abstractmethod
    @overrides
    def __init__(self, **kwargs):
        super(DocumentCleaner, self).__init__(**kwargs)


class RemoveStopWord(DocumentCleaner):
    """Remove stop words."""

    @overrides
    def __init__(self, **kwargs):
        super(RemoveStopWord, self).__init__(**kwargs)

    @overrides
    def process(self, clao_info: TextCLAO, stopwords=None, replace=False):
        stopwords = stopwords if stopwords else text.ENGLISH_STOP_WORDS
        replace_token = STOPWORD if replace else ''
        pattern = r'\b(' + '|'.join(stopwords) + r')\b'
        if not replace:
            pattern = r'\s?' + pattern

        stopword_pattern = re.compile(pattern, flags=re.IGNORECASE)
        raw_text = clao_info.annotations.elements[RAW_TEXT].raw_text
        clao_info.annotations.elements[RAW_TEXT].raw_text = stopword_pattern.sub(replace_token, raw_text)


class ConvertToLowerCase(DocumentCleaner):
    """Converts all letters in the raw text into lower case."""

    @overrides
    def __init__(self, **kwargs):
        super(ConvertToLowerCase, self).__init__(**kwargs)

    @overrides
    def process(self, clao_info: TextCLAO):
        raw_text = clao_info.annotations.elements[RAW_TEXT].raw_text
        clao_info.annotations.elements[RAW_TEXT].raw_text = raw_text.lower()


class ExcludePunctuation(DocumentCleaner):
    """Excludes punctuations from the raw text."""

    @overrides
    def __init__(self, **kwargs):
        super(ExcludePunctuation, self).__init__(**kwargs)

    @overrides
    def process(self, clao_info: TextCLAO):
        punctuations = '.,!:;'
        raw_text = clao_info.annotations.elements[RAW_TEXT].raw_text
        clao_info.annotations.elements[RAW_TEXT].raw_text = re.sub("[" + re.escape(punctuations) + "]", '', raw_text)


# class Lemmatization(DocumentCleaner):
#     """Remove inflectional endings only and return the base or dictionary form of a word."""
#     @abstractmethod
#     @overrides
#     def __init__(self, **kwargs):
#         super(Lemmatization, self).__init__(**kwargs)
#
#     @overrides
#     def process(self, clao_info: TextCLAO):
#         #TODO: update Lemmatization using function from SpaCy
#         lemmatized_sentence = " ".join([lemma(word)
#                                         for word in clao_info.annotations.elements[RAW_TEXT].raw_text.split()])
#         return lemmatized_sentence


# class Stem(DocumentCleaner):
#     """Reduce a word to its word stem that affixes to suffixes and prefixes
#     or to the roots of words known as a lemma."""
#
#     @abstractmethod
#     @overrides
#     def __init__(self, **kwargs):
#         super(Stem, self).__init__(**kwargs)
#
#     @overrides
#     def process(self, clao_info: TextCLAO):
#         sno = nltk.stem.SnowballStemmer('english')
#         stemmed_sentence = " ".join([sno.stem(word)
#                                      for word in clao_info.annotations.elements[RAW_TEXT].raw_text.split()])
#         return stemmed_sentence


class SpellChecker(DocumentCleaner):
    """Check spells and corrects.

    Reference:
    1. Peter Norvig’s spell checker implementation (​http://www.norvig.com/spell-correct.html​)
    2. MedCat: A word is spelled against the VCB, but corrected only against the CDB.
    3. Others
    """

    @abstractmethod
    @overrides
    def __init__(self, **kwargs):
        super(SpellChecker, self).__init__(**kwargs)

    @overrides
    def process(self, clao_info: TextCLAO):
        # TODO: add one method
        pass
