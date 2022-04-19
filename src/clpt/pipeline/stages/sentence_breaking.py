"""NLP pipeline stage for splitting the text into sentences."""

import re
from abc import abstractmethod
from typing import List

from src.clpt.pipeline.stages.pipeline_stage import PipelineStage



class SentenceBreaking(PipelineStage):
    """Abstract sentence breaking class. Splits sentences using a regular expression or space.
    """
    def __init__(self):
        pass

