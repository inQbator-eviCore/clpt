import logging
from abc import abstractmethod

import numpy as np
from blist import blist
from gensim.models.keyedvectors import KeyedVectors

from src.clao.text_clao import Embedding, TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import EMBEDDINGS, SENTENCES, TOKENS

OOV = '<OOV>'

logger = logging.getLogger(__name__)

# TODO Add documentation to this module


class EmbeddingsStage(PipelineStage):
    def __init__(self, embeddings_file_path, binary: bool = True, **kwargs):
        super(EmbeddingsStage, self).__init__(**kwargs)
        if embeddings_file_path:
            self.model = KeyedVectors.load_word2vec_format(embeddings_file_path, binary=binary)

    @abstractmethod
    def process(self, clao_info: TextCLAO) -> None:
        pass


class SimpleWordEmbeddings(EmbeddingsStage):
    def __init__(self, **kwargs):
        super(SimpleWordEmbeddings, self).__init__(**kwargs)

    def process(self, clao_info: TextCLAO) -> None:
        vector_size = self.model.vector_size
        embeddings = blist()

        # Add random normal vector as OOV placeholder
        embeddings.append(Embedding(0, np.random.normal(0, 0.1, vector_size)))
        key_to_embedding_id = {OOV: 0}

        for token in clao_info.get_annotations(TOKENS):
            key = token.text
            if key not in key_to_embedding_id:
                try:
                    embedding = Embedding(len(embeddings), self.model.get_vector(key))
                    embeddings.append(embedding)
                    key_to_embedding_id[key] = embedding.element_id
                except KeyError:
                    logger.debug(f"Key {key} not found in embeddings.")
                    key = OOV
            token._embedding_id = key_to_embedding_id[key]
        clao_info.insert_annotations(EMBEDDINGS, embeddings)


class SimpleSentenceEmbeddings(EmbeddingsStage):
    def __init__(self):
        super(SimpleSentenceEmbeddings, self).__init__(embeddings_file_path=None)

    def process(self, clao_info: TextCLAO) -> None:
        embeddings = clao_info.get_annotations(EMBEDDINGS)
        embedding_id_offset = len(embeddings)
        sent_embeddings = blist()
        for i, sent in enumerate(clao_info.get_annotations(SENTENCES)):
            vectors = [token.embedding.vector for token in sent.tokens]
            sent_vector = np.mean(vectors, axis=0)
            embedding = Embedding(embedding_id_offset + i, sent_vector)
            sent_embeddings.append(embedding)
            sent._embedding_id = embedding.element_id
        clao_info.insert_annotations(EMBEDDINGS, sent_embeddings)
