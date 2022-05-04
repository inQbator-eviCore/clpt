from abc import ABC

import spacy

from src.clao.text_clao import Sentence, TextCLAO, Token
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.clpt.pipeline.stages.tokenization import logger
from src.constants.annotation_constants import LEMMA, POS, SENTENCES, TOKENS


class SpaCyStage(PipelineStage, ABC):
    def __init__(self, timeout_seconds=1, disable=None, **kwargs):
        """add docstring here"""
        super(SpaCyStage, self).__init__(timeout_seconds, **kwargs)

        # Other ways to download spacy 'en' model:
        #  1) `python -m spacy download en_core_web_sm`
        #  2) download from https://github.com/explosion/spacy-models/releases?q=en_core_web_sm&expanded=true
        #  and copy the artifacts to the repo and execute `pip install en_core_web_sm-3.3.0.tar.gz` or other version of
        # the model

        if not disable:
            disable = []
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=disable)
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm', disable=disable)


class SpaCyProcessing(SpaCyStage):
    def __init__(self, pos: bool = False, lemma: bool = False, break_sentences: bool = False, **kwargs):
        """add docstring here"""
        disable = ['ner']
        if (lemma or break_sentences) and not pos:
            logger.info('lemmatization and sentence breaking require part-of-speech tags. Enabling POS for this run')
            pos = True
        if not pos:
            disable.append('tagger')
        if not lemma:
            disable.append('lemmatizer')
        if not break_sentences:
            disable.append('senter')
        self.break_sentences = break_sentences
        super(SpaCyProcessing, self).__init__(disable=disable, **kwargs)

    def process(self, clao_info: TextCLAO) -> None:
        span_text = clao_info.get_text_from(clao_info)
        spacy_doc = self.nlp(span_text)

        new_tokens = []
        first_token_id = len(clao_info.get_all_annotations_for_element(TOKENS))
        for i, token in enumerate(spacy_doc):
            element_id = first_token_id + i
            start_offset = clao_info.start_offset + token.idx
            end_offset = start_offset + len(token)
            token_text = span_text[start_offset:end_offset]
            token_map = {}
            if token.pos:
                token_map[POS] = token.pos_
            if token.lemma:
                token_map[LEMMA] = token.lemma_
            new_tokens.append(Token(start_offset, end_offset, element_id, token_text, token_map))
        clao_info.insert_annotations(TOKENS, new_tokens)

        if self.break_sentences:
            new_sentences = []
            first_sentence_id = len(clao_info.get_all_annotations_for_element(SENTENCES))
            for i, sent in enumerate(spacy_doc.sents):
                start_offset = sent.start_char
                end_offset = sent.end_char
                element_id = first_sentence_id + i
                start_token_id = sent.start + first_token_id
                end_token_id = sent.end + first_token_id
                new_sentences.append(Sentence(start_offset, end_offset, element_id, clao_info, None,
                                              (start_token_id, end_token_id)))
            clao_info.insert_annotations(SENTENCES, new_sentences)
