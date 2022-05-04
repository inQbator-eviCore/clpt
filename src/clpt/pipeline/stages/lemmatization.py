"""Add the corresponding stem to each tokens."""
from abc import abstractmethod

import nltk
from nltk.stem import WordNetLemmatizer
from overrides import overrides

from src.clao.text_clao import TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.clpt.pipeline.stages.spacy_processing import SpaCyStage
from src.constants.annotation_constants import LEMMA, TOKENS


class Lemmatization(PipelineStage):
    @abstractmethod
    @overrides
    def __init__(self, **kwargs):
        super(Lemmatization, self).__init__(**kwargs)


class WordnetLemma(Lemmatization):
    """Reduce a word tothe roots of words known as a lemma by NLTK WordNetLemmatizer."""
    @overrides
    def __init__(self, **kwargs):
        super(WordnetLemma, self).__init__(**kwargs)
        try:
            self.lemmatizer = WordNetLemmatizer()
        except OSError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            self.lemmatizer = WordNetLemmatizer()

    @overrides
    def process(self, clao_info: TextCLAO):
        for token in clao_info.get_all_annotations_for_element(TOKENS):
            token.map[LEMMA] = self.lemmatizer.lemmatize(token.text)


class SpaCyLemma(SpaCyStage, Lemmatization):
    """Reduce a word tothe roots of words known as a lemma by spaCy."""

    @overrides
    def __init__(self, **kwargs):
        super(SpaCyLemma, self).__init__(disable=['parser', 'ner'], **kwargs)

    @overrides
    def process(self, clao_info: TextCLAO):
        for token in clao_info.get_all_annotations_for_element(TOKENS):
            for t in self.nlp(token.text):
                token.map[LEMMA] = t.lemma_
