import logging
from typing import List
import importlib
from sklearn.model_selection import train_test_split
from src.clao.text_clao import PredictProbabilities, ActualDataSource, \
    TextCLAO, Predictions
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.utils import extract_vector_from_claos_ml, \
    extract_embedding_from_claos, extract_gold_standard_ds_filter_from_claos, \
    extract_gold_standard_outcome_from_claos
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class MLModel(PipelineStage):
    """ Classification  class to run a sklearn ml model and generate predictions.
        Args:
            model: machine learning model name
            parameters: Model parameters to be passed if any
    """
    def __init__(self, model_name: str, params: str = None, **kwargs):
        super(MLModel, self).__init__(**kwargs)
        self.single_clao = False
        self.model = model_name
        self.parameters = params

    def process(self, claos: List[TextCLAO]) -> None:
        """Perform classification on the data

        Args:
            clao_info: the CLAO information to process
        """

        module_name = 'sklearn.ensemble'
        module = importlib.import_module(module_name)
        model = getattr(module, self.model)
        clf = model(**self.parameters)
        if(claos[0].get_annotations(ActualDataSource)):
            train_embeddings = extract_vector_from_claos_ml(claos, 'train')
            test_embeddings = extract_vector_from_claos_ml(claos, 'test')
            train_labels = extract_gold_standard_ds_filter_from_claos(claos, 'train')
            X_train = np.array(list(train_embeddings.values()))
            X_test = np.array(list(test_embeddings.values()))
            y_train = np.array(list(train_labels.values()))
            clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)
            df = pd.DataFrame(test_embeddings.keys(), columns=['doc_name'])
            df['probab'] = predicted
            if 'probab' in df.columns:
                for clao in claos:
                    try:
                        probability_d = df.loc[df['doc_name'].astype(str) ==
                                               clao.name, 'probab'].item()
                        clao.insert_annotation(Predictions,
                                               Predictions(float(probability_d)))
                    except ValueError as e:
                        logger.info("Execution error at unsupervised")
                        logger.info(e)
                        # TODO Better handling for repeat lines
                        pass
        else:
            embeddings = extract_embedding_from_claos(claos)
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
            ) = train_test_split(u, v, indices, test_size=0.25,
                                 random_state=27, stratify=v)
            clf.fit(X_train, y_train)
            predicted = clf.predict_proba(X_test)[:, 1]
            for v, probab in zip(indices_test, predicted):
                clao = claos[v]
                clao.insert_annotation(PredictProbabilities,
                                       PredictProbabilities(float(probab)))
