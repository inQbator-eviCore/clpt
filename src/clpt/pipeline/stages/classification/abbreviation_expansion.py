"""Expand Abbreviation.

Abbreviations are usually very content specific.

no complete list covering all clinical
abbreviations and their possible senses currently exists.

Many clinical abbreviations are invented in an ad hoc manner by health care providers during practice and hence they are
usually very content/context specific. In the long term, we will need to construct a list covering all clinical
abbreviations and their possible senses related to eviCore data and business.

Currently in this module, we will provide general method for abbreviation expansion.
"""
import logging
from abc import abstractmethod
from overrides import overrides

# AbbreviationDetector is needed to add the abbreviation pipe to the spacy pipeline
from scispacy.abbreviation import AbbreviationDetector  # noqa: F401

from src.clao.text_clao import TextCLAO, Token, Text
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.clpt.pipeline.stages.analysis.spacy_processing import SpaCyStage
from src.constants.annotation_constants import CLEANED_TEXT, RAW_TEXT, ABBR_DICT
from src.constants.annotation_constants import EXPAND_ABBREVIATION


logger = logging.getLogger(__name__)


class AbbreviationExpansion(PipelineStage):
    """Expand abbreviated tokens to their full form"""

    @abstractmethod
    @overrides
    def __init__(self, **kwargs):
        super(AbbreviationExpansion, self).__init__(**kwargs)


class AbbreviationExpandWithDict(AbbreviationExpansion):
    """Expand the abbreviation of raw texts in CLAO using the user provided dictionary."""
    @overrides
    def __init__(self, abbreviation_dict=ABBR_DICT, **kwargs):
        super(AbbreviationExpandWithDict, self).__init__(**kwargs)
        self.abbreviation_dict = abbreviation_dict

    @overrides
    def process(self, clao_info: TextCLAO):
        for token in clao_info.get_annotations(Token):
            if token.text in self.abbreviation_dict:
                token.map[EXPAND_ABBREVIATION] = self.abbreviation_dict[token.text]


class SpacyAbbreviationExpand(SpaCyStage):
    """Expand the abbreviation of cleaned texts or raw texts in CLAO using Scispacy."""

    @overrides
    def __init__(self, **kwargs):
        super(SpacyAbbreviationExpand, self).__init__(**kwargs)
        # Add the abbreviation pipe to the spacy pipeline
        self.nlp.add_pipe("abbreviation_detector")

    @overrides
    def process(self, clao_info: TextCLAO) -> None:
        """Expand abbreviated tokens to their full form adding Scispacy abbreviation detector to Spacy pipeline.

        Args:
            clao_info (TextCLAO): The CLAO information to process
        """
        text_obj = clao_info.get_annotations(Text, {'description': CLEANED_TEXT})
        if text_obj:
            text = text_obj.raw_text
        else:
            text_obj = clao_info.get_annotations(Text, {'description': RAW_TEXT})
            text = text_obj.raw_text

        expanded_text = self.nlp(text)

        # TODO we need to figure out a way to do this without the nested for-loops
        for abrv in expanded_text._.abbreviations:
            for token in clao_info.get_annotations(Token):
                # Because CLPT consider punctuation as an entity, so the start and end index will not match with reuslts
                # returned by spacy abbreviation expansion.
                # TODO: once update the CLPT to not count punctuation as an entity, we could use the start and end index
                #  to add expanded abbreviation (see below)
                # if (abrv.start == token.start_offset) and (abrv.end == token.end_offset):
                #     token.map[EXPAND_ABBREVIATION] = abrv._.long_form

                # For now, will use a simple regex match to add the token
                if abrv.text == token.text:
                    token.map[EXPAND_ABBREVIATION] = abrv._.long_form

        logger.info("Abbreviation Expansion by adding the abbreviation pipe to the spacy pipeline.")
