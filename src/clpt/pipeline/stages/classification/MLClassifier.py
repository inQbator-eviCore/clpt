import logging
from typing import List
# from sklearn.linear_model import SGDClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.clao.text_clao import PredictProbabilities, TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.utils import extract_gold_standard_outcome_from_claos, extract_vector_from_claos
import numpy as np

logger = logging.getLogger(__name__)


class ML_Model(PipelineStage):
    """ Classification  class to run a sklearn ml model and generate predictions.

        Args:
            model: machine learning model name
            parameters: Model parameters to be passed if any
        """
    def __init__(self, model_name: str, params: str = None, **kwargs):
        super(ML_Model, self).__init__(**kwargs)
        self.single_clao = False
        self.model = model_name
        self.parameters = params

    def process(self, claos: List[TextCLAO]) -> None:
        """Perform classification on the data

        Args:
            clao_info: the CLAO information to process
        """

        model_name = self.model+"("+self.parameters+")"
        clf = eval(model_name)
        embeddings = extract_vector_from_claos(claos)
        actual_label_from_claos = extract_gold_standard_outcome_from_claos(claos)
        u = np.array(list(embeddings.values()))
        v = np.array(list(actual_label_from_claos.values()))
        indices = np.arange(v.shape[0])
        (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(u, v, indices, test_size=0.25, random_state=27, stratify=v)
        clf.fit(X_train, y_train)
        predicted = clf.predict_proba(X_test)[:, 1]
        for v, probab in zip(indices_test, predicted):
            clao = claos[v]
            clao.insert_annotation(PredictProbabilities, PredictProbabilities(float(probab)))
