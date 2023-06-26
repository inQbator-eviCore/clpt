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
        if model_name == "Denial_Rules":
            predictions = {}
            text = (clao.get_annotations(Text, {'description': CLEANED_TEXT})).raw_text
            if('urolift' in text):
                predictions['Treatment_Text_pred'] = 'UroLift'
            elif('rezum' in text):
                predictions['Treatment_Text_pred'] = 'Rezum'
            elif('AquaBeam' in text):
                predictions['Treatment_Text_pred'] = 'AquaBeam'
            else:
                predictions['Treatment_Text_pred'] = 'No_Text'
            if(('considered non-standard therapy' in text) or ('experimental' in text)
                    or ('investigational' in text) or 'unproven' in text):
                predictions['experimental_pred'] = 1
            else:
                predictions['experimental_pred'] = 0
            if(('in the absence of sufficient clinical information to support this request' in text)):
                predictions['insufficient_info_pred'] = 1
            else:
                predictions['insufficient_info_pred'] = 0
            if('-the results of a cystoscopy were not included in the records' in text):
                predictions['no_cystoscopy_pred'] = 1
            else:
                predictions['no_cystoscopy_pred'] = 0
            if('-the results of the estimated prostate volume were not included in the records' in text):
                predictions['no_volume_pred'] = 1
            else:
                predictions['no_volume_pred'] = 0
            if('estimated prostate volume was more than 80 cc' in text):
                predictions['volume_gt_80_pred'] = 1
            else:
                predictions['volume_gt_80_pred'] = 0
            if(('not show you had failure' in text) and ('contraindication' in text)
                    and ('intolerance' in text)):
                predictions['no_pills_pred'] = 1
            else:
                predictions['no_pills_pred'] = 0
            if('-you are younger than 50 years' in text):
                predictions['age_lt_50_pred'] = 1
            else:
                predictions['age_lt_50_pred'] = 0
            if('-you are younger than 45 years' in text):
                predictions['age_lt_45_pred'] = 1
            else:
                predictions['age_lt_45_pred'] = 0
            clao.insert_annotation(PredictionsMultiLabels, PredictionsMultiLabels(predictions))
