"""NLP pipeline stage for tokenizing sentences."""

import re
from abc import abstractmethod

from blist import blist
from overrides import overrides

from src.clao.text_clao import IdSpan, Sentence, Span, Token
from src.clao.text_clao import TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import RAW_TEXT, SENTENCES, TOKENS
from src.constants.regex_constants import BIGRAM_FOUR_DIGIT_YEAR_MONTH, BIGRAM_TEXT_MONTH_DAY, BIGRAM_TEXT_MONTH_YEAR, \
    BIGRAM_TEXT_YEAR_MONTH, CONTRACTION, DATE, DOT_JOINED_NUMBER, HYPHENATED_TOKEN, ICD10_CODE, INITIAL, LONG_NUMBER, \
    NUMBER, PERSONAL_TITLE, TIME_12HR, TIME_24HR, TRIGRAM_QUADGRAM_DATE_DATETIME, UNIGRAM_MONTH, UNIGRAM_MONTH_YEAR, \
    UNIGRAM_YEAR, UNIT_WITH_SLASH
from src.utils import match


class Tokenization(PipelineStage):
    """Abstract tokenization stage"""

    @abstractmethod
    @overrides
    def __init__(self, timeout_seconds=5):
        super(Tokenization, self).__init__(timeout_seconds)

    def process(self, clao_info: TextCLAO) -> None:
        raw_text = clao_info.get_all_annotations_for_element(RAW_TEXT).raw_text
        if len(clao_info.get_all_annotations_for_element(SENTENCES)) == 0:
            clao_info.insert_annotation(SENTENCES, Sentence(0, len(raw_text), 0, clao_info))
        for sentence in clao_info.get_all_annotations_for_element(SENTENCES):
            tokens = self.get_tokens(clao_info, sentence)
            clao_info.insert_annotations(TOKENS, tokens)
            sentence._token_id_range = (tokens[0].element_id, tokens[-1].element_id + 1)

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

    TOKENIZATION_REGEXES = ["(?<=^)(?<=\\W)" + ICD10_CODE + "(?=\\W|$)",
                            LONG_NUMBER,
                            PERSONAL_TITLE,
                            CONTRACTION,
                            INITIAL,
                            DOT_JOINED_NUMBER,
                            UNIT_WITH_SLASH,
                            TRIGRAM_QUADGRAM_DATE_DATETIME,
                            BIGRAM_FOUR_DIGIT_YEAR_MONTH,
                            BIGRAM_TEXT_MONTH_YEAR,
                            BIGRAM_TEXT_YEAR_MONTH,
                            BIGRAM_TEXT_MONTH_DAY,
                            UNIGRAM_MONTH_YEAR,
                            UNIGRAM_MONTH,
                            UNIGRAM_YEAR,
                            DATE,
                            TIME_12HR,
                            TIME_24HR,
                            HYPHENATED_TOKEN,
                            NUMBER,
                            "(\\w+)"]

    TOKEN_REGEX_MATCH_STRING = '|'.join(TOKENIZATION_REGEXES)

    def __init__(self, **kwargs):
        super(RegexTokenization, self).__init__(**kwargs)
        self.token_regex = re.compile(self.TOKEN_REGEX_MATCH_STRING)

    def get_tokens(self, clao_info: TextCLAO, span: Span):
        # TODO: Bring this more in line with nlp_pipeline
        token_spans = match(self.token_regex, span.get_text_from(clao_info.annotations), span.start_offset, True)
        token_id_offset = len(clao_info.get_all_annotations_for_element(TOKENS))
        tokens = blist()
        for s in token_spans:
            token_id = token_id_offset + len(tokens)
            text = s.get_text_from(clao_info.annotations)
            t = Token.from_id_span(IdSpan.from_span(s, token_id), text)
            if not self.token_regex.match(text):
                for i, ch in enumerate(text):
                    if not ch.isspace():
                        start = t.start_offset + i
                        end = t.start_offset + i + 1
                        tokens.append(Token(start, end, token_id, text, {}))
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
