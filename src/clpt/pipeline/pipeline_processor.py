from typing import List

from omegaconf import DictConfig

from src.clao.clao import TextCLAO
from src.clpt.pipeline import pipeline_stage_creator
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage


class NlpPipelineProcessor:
    """Class handling the processing of CLAO through the NLP pipeline.

    Attributes:
        pipeline_stages: The PipelineStages to run CLAO through.
    """

    def __init__(self, stages: List[PipelineStage]):
        self.pipeline_stages = stages

    @classmethod
    def from_stages_config(cls, cfg: DictConfig):
        return cls(pipeline_stage_creator.build_pipeline_stages(cfg))

    def process(self, clao_info: TextCLAO) -> None:
        """
        Process documents through the NLP pipeline, editing the document objects in-place
        Args:
            clao_info: The CLAO information to process.
        Returns: None
        """
        for stage in self.pipeline_stages:
            stage.process(clao_info)
