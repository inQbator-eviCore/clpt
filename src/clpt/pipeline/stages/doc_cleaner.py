"""NLP DocumentCleaner stage for cleaning the CLAO.

DocumentCleaner includes removing stop words, converting to lower case, excluding punctuations and checking spelling.
"""
import re
from abc import abstractmethod

from overrides import overrides
from sklearn.feature_extraction import text

from src.clao.text_clao import RawText, TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import CLEANED_TEXT, RAW_TEXT

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
        raw_text_obj = clao_info.get_annotations(RAW_TEXT)
        raw_text = raw_text_obj.raw_text
        raw_text_obj.raw_text = stopword_pattern.sub(replace_token, raw_text)


class ConvertToLowerCase(DocumentCleaner):
    """Converts all letters in the raw text into lower case."""

    @overrides
    def __init__(self, **kwargs):
        super(ConvertToLowerCase, self).__init__(**kwargs)

    @overrides
    def process(self, clao_info: TextCLAO):
        raw_text_obj = clao_info.get_annotations(RAW_TEXT)
        raw_text = raw_text_obj.raw_text
        raw_text_obj.raw_text = raw_text.lower()


class ExcludePunctuation(DocumentCleaner):
    """Excludes punctuations from the raw text."""

    @overrides
    def __init__(self, **kwargs):
        super(ExcludePunctuation, self).__init__(**kwargs)

    @overrides
    def process(self, clao_info: TextCLAO):
        punctuations = '.,!:;'
        raw_text_obj = clao_info.get_annotations(RAW_TEXT)
        raw_text = raw_text_obj.raw_text
        raw_text_obj.raw_text = re.sub("[" + re.escape(punctuations) + "]", '', raw_text)


class DoNothingDocCleaner(DocumentCleaner):
    """DocCleaner that just created CLEANED_TEXT directly from RAW_TEXT. Useful for testing/debugging purposes"""
    def __init__(self, **kwargs):
        super(DoNothingDocCleaner, self).__init__(**kwargs)

    def process(self, clao_info: TextCLAO) -> None:
        raw_text = clao_info.get_annotations(RAW_TEXT).raw_text
        clao_info.insert_annotation(CLEANED_TEXT, RawText(raw_text), element_type_is_list=False)
