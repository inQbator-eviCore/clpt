"""Takes in stages listed in the configuration file and process CLAO(s) using NlpPipelineProcessor."""
from typing import List

from omegaconf import DictConfig

from src.clao.text_clao import TextCLAO
from src.clpt.pipeline import pipeline_stage_creator
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage


class NlpPipelineProcessor:
    """Process CLAO(s) through the NLP pipeline."""
    def __init__(self, single_clao_stages: List[PipelineStage], all_claos_stages: List[PipelineStage]):
        """Class handling the processing of CLAO(s) through the NLP pipeline.

        Args:
            single_clao_stages: PipelineStages to run a single CLAO through at a time.
            all_claos_stages: PipelineStages that run over a series of CLAOs
        """
        self.single_clao_pipeline_stages = single_clao_stages
        self.all_claos_pipeline_stages = all_claos_stages

    @classmethod
    def from_stages_config(cls, cfg: DictConfig):
        """Build pipeline stages."""
        return cls(*pipeline_stage_creator.build_pipeline_stages(cfg))

    def process(self, clao_info: TextCLAO) -> None:
        """Process documents through the NLP pipeline, editing the document objects in-place.

        Args:
            clao_info: The CLAO information to process.

        Returns:
            None
        """
        for stage in [ps for ps in self.single_clao_pipeline_stages]:
            stage.process(clao_info)

    def process_multiple(self, claos: List[TextCLAO]):
        """Process documents through the NLP pipeline for multiple CLAOs

        Args:
            claos: a list of CLAOs to process.

        Returns:
            None
        """
        for stage in [ps for ps in self.all_claos_pipeline_stages]:
            stage.process(claos)
