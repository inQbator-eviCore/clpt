"""NLP pipeline stage for splitting the text into sentences."""

from src.clao.clao import TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage


class SentenceBreaking(PipelineStage):
    """Abstract sentence breaking class. Splits sentences using a regular expression or space.
    """
    def __init__(self):
        pass

    def process(self, clao_info: TextCLAO) -> None:
        pass
