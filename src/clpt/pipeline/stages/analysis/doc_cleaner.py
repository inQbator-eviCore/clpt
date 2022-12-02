"""NLP DocumentCleaner stage for cleaning the CLAO.

DocumentCleaner includes removing stop words, converting to lower case, and excluding punctuations.
"""
import re
from abc import abstractmethod

from overrides import overrides
from sklearn.feature_extraction import text
from src.clao.text_clao import Text, TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import CLEANED_TEXT, RAW_TEXT

STOPWORD = '<stopword>'


class DocumentCleaner(PipelineStage):
    """Clean the raw texts in CLAO(s) and add the cleaned text to CLAO(s)."""
    @abstractmethod
    @overrides
    def __init__(self, **kwargs):
        super(DocumentCleaner, self).__init__(**kwargs)

    def process(self, clao_info: TextCLAO) -> None:
        """Clean the raw texts in CLAO(s) and add the cleaned text to CLAO(s).

        Args:
            clao_info (TextCLAO): The CLAO information to process
        """
        text_obj = clao_info.get_annotations(Text, {'description': CLEANED_TEXT})
        if text_obj:
            text_obj.raw_text = self.clean_text(text_obj.raw_text)
        else:
            text_obj = clao_info.get_annotations(Text, {'description': RAW_TEXT})
            cleaned_text = self.clean_text(text_obj.raw_text)
            clao_info.insert_annotation(Text, Text(cleaned_text, CLEANED_TEXT))

    @abstractmethod
    def clean_text(self, raw_text: str) -> str:
        """Clean the raw texts in CLAO(s).

        Args:
            raw_text: the raw text from CLAO(s)
        """
        pass


class RemoveStopWord(DocumentCleaner):
    """Remove stop words."""

    @overrides
    def __init__(self, stopwords=None, replace=False, **kwargs):
        """Remove stop words from text in the CLAO(s).

        Args:
            stopwords: a list of stop words to be removed
            replace: if True, replace stop words with ''
        """
        super(RemoveStopWord, self).__init__(**kwargs)

        stopwords = stopwords if stopwords else text.ENGLISH_STOP_WORDS
        pattern = r'\b(' + '|'.join(stopwords) + r')\b'
        if not replace:
            pattern = r'\s?' + pattern
            self.replace_token = ''
        else:
            self.replace_token = STOPWORD

        self.stopword_pattern = re.compile(pattern, flags=re.IGNORECASE)

    @overrides
    def clean_text(self, raw_text: str) -> str:
        """Remove stop words.

        Args:
            raw_text: the raw text from CLAO(s)
        """
        return self.stopword_pattern.sub(self.replace_token, raw_text)


class ConvertToLowerCase(DocumentCleaner):
    """Convert all letters in the text into lower case."""

    @overrides
    def __init__(self, **kwargs):
        super(ConvertToLowerCase, self).__init__(**kwargs)

    @overrides
    def clean_text(self, raw_text: str) -> str:
        """Convert all letters in the text into lower case.

        Args:
            raw_text: the raw text from CLAO(s)
        """
        return raw_text.lower()


class ExcludePunctuation(DocumentCleaner):
    """Exclude punctuations from the text."""

    @overrides
    def __init__(self, **kwargs):
        super(ExcludePunctuation, self).__init__(**kwargs)

    @overrides
    def clean_text(self, raw_text: str) -> str:
        """Exclude punctuations from the text.

        Args:
            raw_text: the raw text from CLAO(s)
        """
        punctuations = '<[^<]+?>'
        raw_text = raw_text.replace('\r', ' ')
        raw_text = raw_text.replace('\n', ' ')
        raw_text = raw_text.replace('\t', ' ')
        return re.sub("[" + re.escape(punctuations) + "]", '', raw_text)


class ExcludeNumbers(DocumentCleaner):
    """Exclude punctuations from the text."""

    @overrides
    def __init__(self, **kwargs):
        super(ExcludeNumbers, self).__init__(**kwargs)

    @overrides
    def clean_text(self, raw_text: str) -> str:
        """Exclude punctuations from the text.

        Args:
            raw_text: the raw text from CLAO(s)
        """
        return re.sub(r'\d+', '', raw_text)


class DoNothingDocCleaner(DocumentCleaner):
    """DocCleaner that just creates CLEANED_TEXT directly from RAW_TEXT.

    Useful for testing/debugging purposes"""
    def __init__(self, **kwargs):
        super(DoNothingDocCleaner, self).__init__(**kwargs)

    def clean_text(self, raw_text: str) -> str:
        """Return the raw text.

        Args:
            raw_text: the raw text from CLAO(s)
        """
        return raw_text
