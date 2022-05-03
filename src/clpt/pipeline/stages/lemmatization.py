"""Add the corresponding stem to each tokens."""
from abc import abstractmethod
from overrides import overrides
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from src.clao.text_clao import TextCLAO
from src.constants.annotation_constants import TOKENS, LEMMA
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage


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


class SpaCyLemma(Lemmatization):
    """Reduce a word tothe roots of words known as a lemma by spaCy."""

    @overrides
    def __init__(self, **kwargs):
        super(SpaCyLemma, self).__init__(**kwargs)
        # Other ways to download spacy 'en' model:
        #  1) `python -m spacy download en_core_web_sm`
        #  2) download from https://github.com/explosion/spacy-models/releases?q=en_core_web_sm&expanded=true
        #  and copy the artifacts to the repo and execute `pip install en_core_web_sm-3.3.0.tar.gz` or other version of
        # the model

        # Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    @overrides
    def process(self, clao_info: TextCLAO):
        for token in clao_info.get_all_annotations_for_element(TOKENS):
            for t in self.nlp(token.text):
                token.map[LEMMA] = t.lemma_
