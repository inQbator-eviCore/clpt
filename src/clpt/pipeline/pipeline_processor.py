from typing import List

from src.clao.clao import ClinicalLanguageAnnotationObject
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
    def from_stages_config(cls, stages_config_path: str):
        return cls(pipeline_stage_creator.build_pipeline_stages(stages_config_path))

    def process(self, clao_info: ClinicalLanguageAnnotationObject) -> None:
        """
        Process documents through the NLP pipeline, editing the document objects in-place
        Args:
            clao_info: The CLAO information to process.
        Returns: None
        """
        for stage in self.pipeline_stages:
            stage.process_with_fallback(clao_info)
