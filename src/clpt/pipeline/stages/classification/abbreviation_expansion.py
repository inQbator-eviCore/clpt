"""NLP DocumentCleaner stage for cleaning the CLAO.

DocumentCleaner includes removing stop words, converting to lower case, and excluding punctuations.
"""
import logging

from src.clao.text_clao import TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage

logger = logging.getLogger(__name__)


class AbbreviationExpansion(PipelineStage):
    """Expand abbreviated tokens to their full form"""

    def __init__(self, **kwargs):
        super(AbbreviationExpansion, self).__init__(**kwargs)

    def process(self, clao_info: TextCLAO) -> None:
        """Expand abbreviated tokens to their full form.

        Args:
            clao_info (TextCLAO): The CLAO information to process
        """
        logger.info("AbbreviationExpansion not yet implemented. Skipping")
