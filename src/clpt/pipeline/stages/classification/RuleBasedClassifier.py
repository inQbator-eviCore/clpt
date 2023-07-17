'''
Rule Based classifier class is intentionally left blank to create 
custom rules based on your needs
'''
import logging
from src.clao.text_clao import PredictionsMultiLabels, Text, TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import CLEANED_TEXT

logger = logging.getLogger(__name__)


class RuleBased(PipelineStage):
    """ Classification  class to run a sklearn ml model and generate predictions.
        Args:
            model: machine learning model name
            parameters: Model parameters to be passed if any
    """
    def __init__(self, model_name: str, params: str = None, **kwargs):
        super(RuleBased, self).__init__(**kwargs)
        self.single_clao = True
        self.model = model_name

    def process(self, clao: TextCLAO) -> None:
        """Perform classification on the data

        Args:
            clao: the CLAO information to process
            
        """
        model_name = self.model
        logger.info("RuleBased engine is intentionally left blank to create \
                     custom rules based on your needs")
