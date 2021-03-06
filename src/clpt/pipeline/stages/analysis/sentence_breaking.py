"""NLP pipeline stage for splitting the text into sentences."""
import re

from blist import blist

from src.clao.text_clao import Sentence, Text, TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import CLEANED_TEXT, RAW_TEXT
from src.constants.regex_constants import SENTENCE_REGEX
from src.utils import match


class SentenceBreaking(PipelineStage):
    """Abstract sentence breaking class. Splits sentences using a regular expression or space."""
    def __init__(self):
        pass

    def process(self, clao_info: TextCLAO) -> None:
        pass


class RegexSentenceBreaking(PipelineStage):
    """Splits sentences using a regular expression."""
    def __init__(self, **kwargs):
        """Load regular expression for sentences breaking."""
        super(RegexSentenceBreaking, self).__init__(**kwargs)
        self.pattern = re.compile(SENTENCE_REGEX)

    def process(self, clao_info: TextCLAO) -> None:
        """Split sentences based on regular expression and add the split sentences into CLAO(s).

        Args:
            clao_info (TextCLAO): the CLAO information to process
        Returns:
            None
        """
        text = (clao_info.get_annotations(Text, {'description': CLEANED_TEXT})
                or clao_info.get_annotations(Text, {'description': RAW_TEXT})).raw_text
        sentence_spans = match(self.pattern, text, clao_info.start_offset, False)
        sentences = clao_info.get_annotations(Sentence)
        sentence_idx_offset = len(sentences)
        new_sentences = blist()
        for i, s in enumerate(sentence_spans):
            element_id = sentence_idx_offset + i
            new_sentences.append(Sentence(s.start_offset, s.end_offset, element_id, clao_info))
        clao_info.insert_annotations(Sentence, new_sentences)
