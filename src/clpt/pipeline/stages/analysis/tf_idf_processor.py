"""NLP pipeline stage for splitting the text into sentences."""
import logging
from typing import List
from blist import blist
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from sklearn.feature_extraction.text import TfidfVectorizer
from src.clao.text_clao import Text, TextCLAO, EmbeddingVector
from src.constants.annotation_constants import CLEANED_TEXT

logger = logging.getLogger(__name__)


class tfidf_vector_processor(PipelineStage):
    """TFIDF class to generate features for a document in matrix format.
    Args:
        vector_file_path: a file which stores the  vector
        binary: if True, the data will be saved in binary format, else it will be saved in plain text.
    """

    def __init__(self, **kwargs):
        super(tfidf_vector_processor, self).__init__(**kwargs)
        self.single_clao = False

    def process(self, claos: List[TextCLAO]) -> None:
        """Add vectors to Text
        Args:
            clao_info (TextCLAO): the CLAO information to process
        Returns:
        None
        """
        corpus = []
        for clao in claos:
            text = (clao.get_annotations(Text, {'description': CLEANED_TEXT})).raw_text
            corpus.append(text)
        vectorizer = TfidfVectorizer(stop_words='english', norm='l2', encoding='latin-1',
                                     max_features=300, ngram_range=(1, 2), min_df=0.1)
        X = vectorizer.fit_transform(corpus)
        vector_embeddings = blist()
        for clao, vector_embeddings in zip(claos, X.toarray()):
            tfidf_vector = EmbeddingVector(vector_embeddings)
            if(vector_embeddings.shape[0] != 0):
                clao.insert_annotation(EmbeddingVector, tfidf_vector)
            else:
                continue
