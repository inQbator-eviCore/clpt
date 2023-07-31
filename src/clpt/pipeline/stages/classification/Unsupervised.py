import logging
from typing import List
from src.clao.text_clao import TextCLAO, Predictions
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.utils import extract_vector_from_claos, extract_embedding_from_claos
import numpy as np
import pandas as pd
from scipy.spatial import distance
import statistics as st

logger = logging.getLogger(__name__)


class EmbeddingDistance(PipelineStage):

    """ Unsupervised Classification using cosine similarity.

        Args:
            model: machine learning model name
            parameters: Model parameters to be passed if any
        """

    def __init__(self, model_name: str, params: str = None, **kwargs):
        super(EmbeddingDistance, self).__init__(**kwargs)
        self.single_clao = False
        self.model = model_name

    def process(self, claos: List[TextCLAO], claos_tax: List[TextCLAO]) -> None:
        """Perform classification on the data
        Args:
            clao_info: the CLAO information to process
        """
        test_embeddings = extract_vector_from_claos(claos, 'test')
        X_test = np.array(list(test_embeddings.values()))
        tax_embed = extract_embedding_from_claos(claos_tax)
        tax_embed_n = np.array(list(tax_embed.values()))
        pred = []
        for x in X_test:
            x_n = np.array(x)
            a = np.array([r for r in x_n if any(r)])
            mins = []
            for y in tax_embed_n:
                y_n = np.array(y)
                b = np.array([r for r in y_n if any(r)])
                cos_distance = distance.cdist(a, b, 'cosine')
                dist_mean = np.mean(cos_distance, axis=1)
                mins.append(dist_mean)
            mins = np.array(mins)
            min_class_sent = st.mode(np.array(mins.argmin(0)))
            val = list(tax_embed.keys())[min_class_sent]
            pred.append(val)
        df = pd.DataFrame(test_embeddings.keys(), columns=['doc_name'])
        df['probab'] = pred
        if 'probab' in df.columns:
            for clao in claos:
                try:
                    probability_d = df.loc[df['doc_name'].astype(str) == clao.name, 'probab'].item()
                    clao.insert_annotation(Predictions, Predictions(float(probability_d)))
                except ValueError as e:
                    logger.info("Execution error at unsupervised")
                    logger.info(e)
                    # TODO Better handling for repeat lines
                    pass
