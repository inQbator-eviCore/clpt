# coding=utf-8
"""NLP pipeline stage for splitting the text into sentences."""

import logging
from typing import List

import nltk
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.clao.text_clao import Text, TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import RAW_TEXT

stop_words = []
try:
    from nltk.corpus import stopwords
except ImportError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords

    stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good',
                   'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather',
                   'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line',
                   'even', 'also', 'may', 'take', 'come'])

logger = logging.getLogger(__name__)


class Cluster(PipelineStage):
    """Class used to generate clusters from a corpus of CLAOs
    Args:
        clustering_method (str): tsne, mds
        output_dir (str): where to output
        n_components (int): number of clusters
        lowercase (bool): whether to lowercase all tokens
        max_df (float): maximum document frequency as a percentage
        min_df (int): minimum document frequency
        ngram_range (tuple): range of ngrams to use
        max_features (int): maximum vocabulary size
    """

    def __init__(self, max_features=5000, ngram_range=(1, 4), min_df=10, max_df=.7, lowercase=True,
                 n_components=10, output_dir='.', clustering_method='tsne', **kwargs):
        """
        """
        super(Cluster, self).__init__(**kwargs)
        self.single_clao = False
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.lowercase = lowercase

        self.n_components = n_components
        self.clustering_method = clustering_method
        self.output_dir = output_dir

    def process(self, claos: List[TextCLAO]) -> None:
        """Create clusters from text
        Args:
            claos (TextCLAO): the CLAO information to process
        Returns:
        None
        """
        corpus = []
        for clao in claos:
            obj = clao.get_annotations(Text, {'description': RAW_TEXT})
            corpus.append(obj.raw_text)

        vectorizer = CountVectorizer(stop_words=stop_words, max_df=self.max_df, min_df=self.min_df,
                                     max_features=self.max_features, ngram_range=self.ngram_range,
                                     lowercase=self.lowercase, strip_accents='unicode')

        dtm_cv = vectorizer.fit_transform(corpus)
        lda_cv = LatentDirichletAllocation(n_components=self.n_components, random_state=0)
        lda_cv.fit(dtm_cv)
        vis_data = pyLDAvis.sklearn.prepare(lda_cv, dtm_cv, vectorizer, mds=self.clustering_method)
        pyLDAvis.save_html(vis_data,
                           f'{self.output_dir}/countvec_ntopics_{self.n_components}_dist_{self.clustering_method}.html')
