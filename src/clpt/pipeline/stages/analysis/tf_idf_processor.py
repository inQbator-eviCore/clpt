"""NLP pipeline stage for splitting the text into sentences."""
import logging
from typing import List
from blist import blist
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
import importlib
from src.clao.text_clao import Text, TextCLAO, EmbeddingVector
from src.constants.annotation_constants import CLEANED_TEXT

logger = logging.getLogger(__name__)


class VectorProcessor(PipelineStage):
    """ Vector class to generate features for a document in matrix format.
    Args:
        vector_file_path: a file which stores the  vector
        binary: if True, the data will be saved in binary format, else it will be saved in plain text.
    """

    def __init__(self, vector_name: str, params: dict = None, **kwargs):
        super(VectorProcessor, self).__init__(**kwargs)
        self.single_clao = False
        self.vector = vector_name
        self.parameters = params

    def process(self, claos: List[TextCLAO]) -> None:
        """Add vectors to Text
        Args:
            clao_info (TextCLAO): the CLAO information to process
        Returns:
        None
        """
        module_name = 'sklearn.feature_extraction.text'
        module = importlib.import_module(module_name)
        model = getattr(module, self.vector)
        model_name = model(**self.parameters)
        model_name = self.vector+"("+self.parameters+")"
        corpus = []
        for clao in claos:
            text = (clao.get_annotations(Text, {'description': CLEANED_TEXT})).raw_text
            corpus.append(text)
        vectorizer = model_name
        X = vectorizer.fit_transform(corpus)
        vector_embeddings = blist()
        for clao, vector_embeddings in zip(claos, X.toarray()):
            tfidf_vector = EmbeddingVector(vector_embeddings)
            if(vector_embeddings.shape[0] != 0):
                clao.insert_annotation(EmbeddingVector, tfidf_vector)
            else:
                continue
