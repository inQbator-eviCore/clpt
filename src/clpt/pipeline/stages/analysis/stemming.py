"""Add the corresponding stem to each tokens."""
from abc import abstractmethod

from nltk.stem import PorterStemmer
from overrides import overrides

from src.clao.text_clao import TextCLAO, Token
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import SPELL_CORRECTED_TOKEN, STEM


class Stemming(PipelineStage):
    """A stemming method from NLTK (PorterStemmer)."""
    @abstractmethod
    @overrides
    def __init__(self, **kwargs):
        super(Stemming, self).__init__(**kwargs)
        self.ps = PorterStemmer('NLTK_EXTENSIONS')


class PorterStemming(Stemming):
    """Reduce a word to its word stem that affixes to suffixes and prefixes."""

    @overrides
    def __init__(self, **kwargs):
        super(PorterStemming, self).__init__(**kwargs)

    @overrides
    def process(self, clao_info: TextCLAO):
        """Reduce a word to its word stem and store the stem to CLAO(s).

        Args:
            clao_info (TextCLAO): the CLAO information to process
        """
        for token in clao_info.get_annotations(Token):
            if SPELL_CORRECTED_TOKEN in token.map:
                token.map[STEM] = self.ps.stem(token.map[SPELL_CORRECTED_TOKEN])
            else:
                token.map[STEM] = self.ps.stem(token.text)
