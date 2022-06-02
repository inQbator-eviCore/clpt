"""Add the corresponding lemmas to each token."""
from abc import abstractmethod

from nltk.stem import WordNetLemmatizer
from overrides import overrides

from src.clao.text_clao import TextCLAO, Token
from src.clpt.pipeline.stages.pipeline_stage import NltkStage, PipelineStage
from src.clpt.pipeline.stages.spacy_processing import SpaCyStage
from src.constants.annotation_constants import LEMMA


class Lemmatization(PipelineStage):
    """Apply lemmatization method to each token."""
    @abstractmethod
    @overrides
    def __init__(self, **kwargs):
        super(Lemmatization, self).__init__(**kwargs)


class WordnetLemma(Lemmatization, NltkStage):
    """Reduce a word to its lemma using NLTK WordNetLemmatizer."""
    @overrides
    def __init__(self, **kwargs):
        super(WordnetLemma, self).__init__(nltk_reqs=['corpora/wordnet', 'corpora/omw-1.4'], **kwargs)
        self.lemmatizer = WordNetLemmatizer()

    @overrides
    def process(self, clao_info: TextCLAO):
        """Use NLTK WordNetLemmatizer for lemmatization and store the lemma to CLAO(s).

        Args:
            clao_info (TextCLAO): the CLAO information to process
        """
        for token in clao_info.get_annotations(Token):
            token.map[LEMMA] = self.lemmatizer.lemmatize(token.text)


class SpaCyLemma(SpaCyStage, Lemmatization):
    """Reduce a word to its lemma using spaCy."""

    @overrides
    def __init__(self, **kwargs):
        super(SpaCyLemma, self).__init__(disable=['parser', 'ner'], **kwargs)

    @overrides
    def process(self, clao_info: TextCLAO):
        """Use lemmatization method in spaCy and store the lemma to CLAO(s).

        Args:
            clao_info (TextCLAO): the CLAO information to process
        """
        for token in clao_info.get_annotations(Token):
            for t in self.nlp(token.text):
                token.map[LEMMA] = t.lemma_
