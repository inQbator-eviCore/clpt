import logging
from abc import abstractmethod
from typing import List

from blist import blist
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors

from src.clao.text_clao import Embedding, Sentence, TextCLAO, Token
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import SPELL_CORRECTED_TOKEN

OOV = '<OOV>'

logger = logging.getLogger(__name__)

# TODO Add documentation to this module


class EmbeddingsStage(PipelineStage):
    def __init__(self, embeddings_file_path, binary: bool = True, **kwargs):
        super(EmbeddingsStage, self).__init__(**kwargs)
        self.vectors: KeyedVectors
        if embeddings_file_path:
            self.vectors = KeyedVectors.load_word2vec_format(embeddings_file_path, binary=binary)

    @abstractmethod
    def process(self, clao_info: TextCLAO) -> None:
        pass


class WordEmbeddings(EmbeddingsStage):
    def __init__(self, replace_oov: bool = True, **kwargs):
        super(WordEmbeddings, self).__init__(**kwargs)
        self.replace_oov = replace_oov

    def process(self, clao_info: TextCLAO) -> None:
        embeddings = blist()

        if self.replace_oov:
            # Add mean of all vectors as OOV placeholder
            embeddings.append(Embedding(0, self.vectors.get_mean_vector(self.vectors.key_to_index.keys())))
            key_to_embedding_id = {OOV: 0}
        else:
            key_to_embedding_id = {}

        for token in clao_info.get_annotations(Token):
            key = token.map.get(SPELL_CORRECTED_TOKEN, token.text)
            if key not in key_to_embedding_id:
                try:
                    embedding = Embedding(len(embeddings), self.vectors.get_vector(key))
                    embeddings.append(embedding)
                    key_to_embedding_id[key] = embedding.element_id
                except KeyError as e:
                    logger.debug(f"Key {key} not found in embeddings.")
                    if self.replace_oov:
                        key = OOV
                    else:
                        raise e
            token._embedding_id = key_to_embedding_id[key]
        clao_info.insert_annotations(Embedding, embeddings)


class FastTextEmbeddings(WordEmbeddings):
    def __init__(self, vector_size, window=3, min_count=1, epochs=10, save_embeddings: bool = False,
                 saved_file_name: str = None, **kwargs):
        """Use Gensim's implementation of FastText* to train embeddings on a corpus of tokenized CLAOs.

        * P. Bojanowski, E. Grave, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information

        Args:
            vector_size: Dimensionality of the word vectors.
            window: The maximum distance between the current and predicted word within a sentence.
            min_count: The model ignores all words with total frequency lower than this.
            epochs: Number of iterations (epochs) over the corpus.
            save_embeddings: Whether or not to save the resulting embeddings to a file.
            saved_file_name: File name to save embeddings to. Must be provided if save_embeddings is True.
            **kwargs: Other arguments of gensim.models.FastText. See official documentation:
                      https://radimrehurek.com/gensim/models/fasttext.html
        """
        super(FastTextEmbeddings, self).__init__(embeddings_file_path=None, replace_oov=False)
        self.model = FastText(vector_size=vector_size, window=window, min_count=min_count, **kwargs)
        self.epochs = epochs

        if save_embeddings and not saved_file_name:
            raise ValueError("Configuration must specify a file name with 'saved_file_name' "
                             "in order to save embeddings.")
        self.save_embeddings = save_embeddings
        self.saved_file_name = saved_file_name

        self.single_clao = False  # TODO handle this much better

    def process(self, claos: List[TextCLAO]) -> None:
        corpus: List[List[str]] = []
        for clao in claos:
            sent_tokens = [[token.map.get(SPELL_CORRECTED_TOKEN, token.text) for token in sent.tokens]
                           for sent in clao.get_annotations(Sentence)]
            corpus.extend(sent_tokens)

        self.model.build_vocab(corpus_iterable=corpus)
        self.model.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=self.epochs)

        self.vectors = self.model.wv
        if self.save_embeddings:
            self.vectors.save_word2vec_format(self.saved_file_name, binary=True)

        for clao in claos:
            super(FastTextEmbeddings, self).process(clao)


class SentenceEmbeddings(EmbeddingsStage):
    """Embeddings can be generated on the fly for any sentence whose tokens have embeddings"""
    def __init__(self):
        super(SentenceEmbeddings, self).__init__(embeddings_file_path=None)

    def process(self, clao_info: TextCLAO) -> None:
        embeddings = clao_info.get_annotations(Embedding)
        embedding_id_offset = len(embeddings)
        sent_embeddings = blist()
        for i, sent in enumerate(clao_info.get_annotations(Sentence)):
            embedding = Embedding(embedding_id_offset + i, sent.get_span_embedding())
            sent_embeddings.append(embedding)
            sent._embedding_id = embedding.element_id
        clao_info.insert_annotations(Embedding, sent_embeddings)
