"""NLP pipeline stage for tokenizing sentences."""

import re
from abc import abstractmethod

from overrides import overrides

from src.clao.annotations import IdSpan, Sentence, Span, Token
from src.clao.clao import TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import RAW_TEXT, SENTENCES
from src.utils import match


class Tokenization(PipelineStage):
    """Abstract tokenization stage"""

    @abstractmethod
    @overrides
    def __init__(self, timeout_seconds=5):
        super(Tokenization, self).__init__(timeout_seconds)

    def process(self, clao_info: TextCLAO) -> None:
        raw_text = clao_info.annotations.elements[RAW_TEXT].raw_text
        if SENTENCES not in clao_info.annotations.elements:
            clao_info.annotations.elements[SENTENCES] = [Sentence(0, len(raw_text), 0)]
        for sentence in clao_info.annotations.elements[SENTENCES]:
            sentence.tokens = self.get_tokens(clao_info, sentence)

    @abstractmethod
    def get_tokens(self, clao_info: TextCLAO, span: Span):
        pass

    @overrides
    def fallback(self, clao_info: TextCLAO) -> None:
        """
        Split on whitespace.
        """
        WhitespaceRegexTokenization().process(clao_info)


class RegexTokenization(Tokenization):
    """Tokenization using regular expressions.

    Attributes:
        token_regex: The regular expression to use for tokenization
    """
    # TODO: use function from SpaCy

    TOKENIZATION_REGEXES = ["(?<=^)(?<=\\W)" + "(?=\\W|$)",
                            "(\\w+)"]

    TOKEN_REGEX_MATCH_STRING = '|'.join(TOKENIZATION_REGEXES)

    def __init__(self, **kwargs):
        super(RegexTokenization, self).__init__(**kwargs)
        self.token_regex = re.compile(self.TOKEN_REGEX_MATCH_STRING)

    def get_tokens(self, clao_info: TextCLAO, span: Span):
        # TODO: Bring this more in line with nlp_pipeline
        token_spans = match(self.token_regex, span.get_text_from(clao_info.annotations), span.start_offset, True)

        tokens = []
        for s in token_spans:
            text = s.get_text_from(clao_info.annotations)
            t = Token.from_id_span(IdSpan.from_span(s, len(tokens)), text)
            if not self.token_regex.match(text):
                for i, ch in enumerate(text):
                    if not ch.isspace():
                        start = t.start_offset + i
                        end = t.start_offset + i + 1
                        tokens.append(Token(start, end, len(tokens), text, {}))
            else:
                tokens.append(t)
        return tokens


class WhitespaceRegexTokenization(RegexTokenization):
    """Tokenization on whitespace (used as a fallback)."""
    TOKENIZATION_REGEXES = '\\S+'
    TOKEN_REGEX_MATCH_STRING = '\\S+'

    def __init__(self, **kwargs):
        super(WhitespaceRegexTokenization, self).__init__(**kwargs)

    def process(self, clao_info: TextCLAO) -> None:
        pass

    def get_tokens(self, clao_info: TextCLAO, span: Span):
        pass
