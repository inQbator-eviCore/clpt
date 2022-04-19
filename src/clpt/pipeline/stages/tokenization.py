"""NLP pipeline stage for tokenizing sentences."""

import re
from abc import abstractmethod
from typing import List

from overrides import overrides
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage


class Tokenization(PipelineStage):
    """Abstract tokenization stage"""

    @abstractmethod
    @overrides
    def __init__(self, timeout_seconds=5):
        super(Tokenization, self).__init__(timeout_seconds)

    @abstractmethod
    def get_tokens(self):
        pass

    @overrides
    def fallback(self) -> None:
        """
        Split on whitespace.
        """


class RegexTokenization(Tokenization):
    """Tokenization using regular expressions.

    Attributes:
        token_regex: The regular expression to use for tokenization
    """
    #TODO: use function from SpaCy

    TOKENIZATION_REGEXES = ["(?<=^)(?<=\\W)" + "(?=\\W|$)",
                            "(\\w+)"]

    TOKEN_REGEX_MATCH_STRING = '|'.join(TOKENIZATION_REGEXES)

    def __init__(self, **kwargs):
        super(RegexTokenization, self).__init__(**kwargs)
        self.token_regex = re.compile(self.TOKEN_REGEX_MATCH_STRING)


class WhitespaceRegexTokenization(RegexTokenization):
    """Tokenization on whitespace (used as a fallback)."""
    TOKENIZATION_REGEXES = '\\S+'
    TOKEN_REGEX_MATCH_STRING = '\\S+'

    def __init__(self, **kwargs):
        super(WhitespaceRegexTokenization, self).__init__(**kwargs)
