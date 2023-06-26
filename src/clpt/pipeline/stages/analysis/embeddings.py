"""NLP Analysis stage for creating s."""
import logging
import numpy as np
from abc import abstractmethod
from typing import List
from blist import blist
from gensim.models import FastText
from gensim.models import Word2Vec
# from gensim.models.keyedvectors import KeyedVectors
from gensim.models import KeyedVectors
import multiprocessing
import sent2vec
from src.clao.text_clao import EmbeddingVector, Sentence, TextCLAO, Token
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
# from src.constants.annotation_constants import SPELL_CORRECTED_TOKEN

OOV = '<OOV>'

logger = logging.getLogger(__name__)


class EmbeddingsStage(PipelineStage):
    """Abstract embedding class with trained embeddings able to be saved in word2vec format.

    Args:
        embeddings_file_path: a file which stores the embedding vector
        binary: if True, the data will be saved in binary word2vec format, else it will be saved in plain text.
    """
    def __init__(self, embeddings_file_path=None, binary: bool = True, **kwargs):
        super(EmbeddingsStage, self).__init__(**kwargs)
        self.vectors: KeyedVectors
        self.binary = binary
        self.embeddings_file_path = embeddings_file_path

    @abstractmethod
    def process(self, claos: List[TextCLAO]) -> None:
        """Process text in CLAO with embedding method."""
        pass


class WordEmbeddings(EmbeddingsStage):
    """Create word embedding(s) using word2vec.

    Attributes:
        replace_oov: if to replace out-of-vocabulary (OOV)

    """
    def __init__(self, embeddings_file_path=None, replace_oov: bool = True, binary: bool = True, **kwargs):
        super(WordEmbeddings, self).__init__(**kwargs)
        self.replace_oov = replace_oov
        self.single_clao = False
        self.embeddings_file_path = embeddings_file_path
        self.binary = binary
        if self.embeddings_file_path:
            logger.info("INITIATE LOADING")
            self.vectors = KeyedVectors.load_word2vec_format(embeddings_file_path, binary=binary)
            logger.info("FINISH LOADING")
            self.dim = len(self.vectors.wv.syn0[0])

    def process(self, claos: List[TextCLAO]) -> None:
        """Process a single tokenized CLAOs to add embeddings for each document
        Args:
            clao_info (TextCLAO): the CLAO information to process
        """
        transformed_X = []
        for clao in claos:
            sent_tokens = [token.text for token in clao.get_annotations(Token)]
            transformed_X.append(np.array(sent_tokens))
        transformed_X_ar = np.array(transformed_X)
        mean_embedding_vectorizer = np.array([
            np.mean([self.vectors[w] for w in words if w in self.vectors]
                    or [np.zeros(self.dim)], axis=0)
            for words in transformed_X_ar
        ])
        vector_embeddings = blist()
        for clao, vector_embeddings in zip(claos, mean_embedding_vectorizer):
            vector = EmbeddingVector(vector_embeddings)
            if(vector_embeddings.shape[0] != 0):
                clao.insert_annotation(EmbeddingVector, vector)
            else:
                continue


class FastTextEmbeddings(WordEmbeddings):
    """FastText Embedding."""
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
        super(FastTextEmbeddings, self).__init__(replace_oov=False)
        self.model = FastText(size=vector_size, window=window, min_count=min_count, **kwargs)
        self.epochs = epochs
        if save_embeddings and not saved_file_name:
            raise ValueError("Configuration must specify a file name with 'saved_file_name' "
                             "in order to save embeddings.")
        self.save_embeddings = save_embeddings
        self.saved_file_name = saved_file_name
        self.single_clao = False  # TODO handle this much better

    def process(self, claos: List[TextCLAO]) -> None:
        """Train embeddings on a corpus of tokenized CLAOs using Gensim's implementation of FastText.

        Args:
            clao_info (TextCLAO): the CLAO information to process
        """
        corpus: List[str] = []
        transformed_X = []
        for clao in claos:
            sent_tokens = [token.text for token in clao.get_annotations(Token)]
            transformed_X.append(np.array(sent_tokens))
            corpus.extend(sent_tokens)
        transformed_X_ar = np.array(transformed_X)
        self.model.build_vocab(corpus, progress_per=10000)
        self.model.train(corpus, total_examples=len(corpus), epochs=self.epochs)
        self.vectors = self.model.wv
        if self.save_embeddings:
            self.vectors.save_word2vec_format(self.saved_file_name, binary=True)
        mean_embedding_vectorizer = np.array([
            np.mean([self.vectors[w] for w in words if w in self.vectors]
                    or [np.zeros(self.dim)], axis=0)
            for words in transformed_X_ar
        ])
        for clao, vector_embeddings in zip(claos, mean_embedding_vectorizer):
            tfidf_vector = EmbeddingVector(vector_embeddings)
            if(vector_embeddings.shape[0] != 0):
                clao.insert_annotation(EmbeddingVector, tfidf_vector)
            else:
                continue


class WordVecEmbeddings(WordEmbeddings):
    """WordVec Embeddings"""
    def __init__(self, min_count=4, window=4, size=300, alpha=0.03, min_alpha=0.0007,
                 sg=1, workers=None, epochs=None, save_embeddings: bool = False,
                 saved_file_name: str = None, **kwargs):

        """Use Gensim's implementation of Word2vec* to train embeddings on a corpus of tokenized CLAOs
        Args:
            min_count: The model ignores all words with total frequency lower than this.
            size: Dimensionality of the word vectors.
            window: The maximum distance between the current and predicted word within a sentence.
            epochs: Number of iterations (epochs) over the corpus.
            save_embeddings: Whether or not to save the resulting embeddings to a file.
            saved_file_name: File name to save embeddings to. Must be provided if save_embeddings is True.
            **kwargs: Other arguments of gensim.models.FastText. See official documentation:
                      https://radimrehurek.com/gensim/models/word2vec.html
        """
        super(WordVecEmbeddings, self).__init__(embeddings_file_path=None, replace_oov=False)
        cores = multiprocessing.cpu_count()
        self.model = Word2Vec(min_count=4, window=4, size=300, alpha=0.03,
                              min_alpha=0.0007, sg=1, workers=cores-1, **kwargs)
        if save_embeddings and not saved_file_name:
            raise ValueError("Configuration must specify a file name with 'saved_file_name' "
                             "in order to save embeddings.")
        self.save_embeddings = save_embeddings
        self.saved_file_name = saved_file_name
        self.epochs = epochs
        self.single_clao = False  # TODO handle this much better

    def process(self, claos: List[TextCLAO]) -> None:
        """Train embeddings on a corpus of tokenized CLAOs using Gensim's implementation of FastText.

        Args:
            clao_info (TextCLAO): the CLAO information to process
        """
        corpus: List[str] = []
        transformed_X = []
        for clao in claos:
            sent_tokens = [token.text for token in clao.get_annotations(Token)]
            transformed_X.append(np.array(sent_tokens))
            corpus.extend(sent_tokens)
        transformed_X_ar = np.array(transformed_X)
        self.model.build_vocab(corpus, progress_per=10000)
        self.model.train(corpus, total_examples=len(corpus), epochs=self.epochs, report_delay=1)
        self.vectors = self.model.wv
        self.dim = len(self.vectors.wv.syn0[0])
        if self.save_embeddings:
            self.vectors.save_word2vec_format(self.saved_file_name, binary=True)
        mean_embedding_vectorizer = np.array([
            np.mean([self.vectors[w] for w in words if w in self.vectors]
                    or [np.zeros(self.dim)], axis=0)
            for words in transformed_X_ar
        ])
        for clao, vector_embeddings in zip(claos, mean_embedding_vectorizer):
            vector = EmbeddingVector(vector_embeddings)
            if(vector_embeddings.shape[0] != 0):
                clao.insert_annotation(EmbeddingVector, vector)
            else:
                continue


class SentenceEmbeddings(EmbeddingsStage):
    """Embeddings can be generated on the fly for any sentence whose tokens have embeddings."""
    def __init__(self, embeddings_file_path=None, replace_oov: bool = True, **kwargs):
        super(SentenceEmbeddings, self).__init__(embeddings_file_path, replace_oov=False)
        self.single_clao = True
        self.model = sent2vec.Sent2vecModel()
        try:
            logger.info(embeddings_file_path)
            self.model.load_model(embeddings_file_path)
            logger.info('model successfully loaded')
        except Exception as e:
            logger.info(e)

    def process(self, clao: TextCLAO) -> None:
        """Create sentence embeddings for any CLAO with Sentences and Embeddings for Tokens.

        Args:
            clao_info (TextCLAO): the CLAO information to process
        """
        sent_embeddings = blist()
        for i, sent in enumerate(clao.get_annotations(Sentence)):
            sent2 = ' '.join([token.text for token in sent.tokens])
            embedding = self.model.embed_sentence(sent2).flatten()
            sent_embeddings.append(embedding)
        clao.insert_annotation(EmbeddingVector, EmbeddingVector(sent_embeddings))
