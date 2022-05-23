import nltk

from src.clao.text_clao import Sentence, TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import NltkStage
from src.constants.annotation_constants import POS


class SimplePOSTagger(NltkStage):
    def __init__(self, **kwargs):
        super(SimplePOSTagger, self).__init__(nltk_reqs=['taggers/averaged_perceptron_tagger'], **kwargs)

    def process(self, clao_info: TextCLAO) -> None:
        for sent in clao_info.get_annotations(Sentence):
            tokens = sent.tokens
            token_texts = [t.text for t in tokens]
            _, tags = zip(*nltk.pos_tag(token_texts))
            for token, pos_tag in zip(tokens, tags):
                token.map[POS] = pos_tag
