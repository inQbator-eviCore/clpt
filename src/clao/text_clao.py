"""Initial thoughts / rough draft of some internal text annotation objects, in line with XML schema illustrated in 2022
NaaCL paper"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from blist import blist
from lxml import etree
from omegaconf import DictConfig

from src.clao.clao import CLAOElement, CLAOElementContainer, ClinicalLanguageAnnotationObject, IdCLAOElement
from src.constants.annotation_constants import ANNOTATION, EMBEDDING, EMBEDDINGS, EMBEDDING_ID, ENTITIES, ENTITY, \
    HEADING, HEADINGS, PARAGRAPH, PARAGRAPHS, RAW_TEXT, SECTION, SENTENCE, SENTENCES, SPAN, TEXT, TOKEN, TOKENS, \
    VECTOR


class Span(CLAOElement):
    """CLAOElement representing a span of text"""
    element_name = SPAN

    def __init__(self, start_offset: int, end_offset: int, span_map=None, *args, **kwargs):
        super(Span, self).__init__(*args, **kwargs)
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.map = {} if span_map is None else span_map

    def adjust_offsets(self, delta: int) -> None:
        """
        Adjust the start and end offsets of this Span by the provided delta
        Args:
            delta: How much to shift the offsets. To reduce the offsets, this should be negative.

        Returns: None
        """
        self.start_offset += delta
        self.end_offset += delta

    def to_json(self) -> Dict:
        return {**super(Span, self).to_json(),
                'start': str(self.start_offset),
                'end': str(self.end_offset),
                **self.map}

    def get_text_from(self, text_clao: 'TextCLAO') -> str:
        """
        Get the text for this Span from the TextCLAO that contains it.
        Args:
            text_clao: The TextCLAO object that contains this Span
        Returns:
            The text encompassed by this Span
        """
        return text_clao.get_text_for_span(self)

    def overlaps(self, other):
        """Determine if this Span overlaps another
        Args:
            other: a Span
        Returns:
            Whether or not these two spans overlap at all
        """
        return self.start_offset < other.end_offset and self.end_offset > other.start_offset

    def contains(self, other):
        """Determine whether this Span contains another
        Args:
            other: a Span
        Returns:
            Whether or not `other` is wholly contained within this Span
        """
        return self.start_offset <= other.start_offset and self.end_offset >= other.end_offset

    def __len__(self):
        return self.end_offset - self.start_offset

    def __eq__(self, other):
        return self.start_offset == other.start_offset and self.end_offset == other.end_offset and self.map == other.map

    def __str__(self):
        return f'Span({self.start_offset}, {self.end_offset})'

    def __repr__(self):
        return str(self)


class TextCLAO(Span, ClinicalLanguageAnnotationObject[str]):
    """CLAO object for handling textual data

    Class Attributes:
        element_name: Name of element as it should appear in an XML tag
        _top_level_elements: List of elements to be serialized in the top level of the annotation schema. All other CLAO
                             elements to be serialized should be contained within one of these
    """
    element_name = ANNOTATION
    _top_level_elements = [RAW_TEXT, SENTENCES, EMBEDDINGS]

    def __init__(self, raw_text: str, name: str, cfg: DictConfig = None, *args, **kwargs):
        super(TextCLAO, self).__init__(start_offset=0, end_offset=len(raw_text), raw_data=RawText(raw_text), name=name,
                                       cfg=cfg, *args, **kwargs)

    @classmethod
    def from_file(cls, input_path: str):
        name = os.path.splitext(os.path.basename(input_path))[0]
        with open(input_path, 'r') as f:
            return cls(f.read(), name)

    def _search_by_val(self, element_type: str, value: str):
        raise NotImplementedError()

    def to_json(self) -> Dict:
        json_dict = super(TextCLAO, self).to_json()
        for element_type in self._top_level_elements:
            if element_type in self.elements:
                element_value = self.elements[element_type]
                if isinstance(element_value, (list, blist)):
                    json_dict[element_type] = [e.to_json() for e in element_value]
                else:
                    json_dict.update(element_value.to_json())
        return json_dict

    def to_xml(self) -> etree.Element:
        annotation = super(TextCLAO, self).to_xml(parent=None, attribs={})
        for element_type in self._top_level_elements:
            if element_type in self.elements:
                element_value = self.elements[element_type]
                if isinstance(element_value, (list, blist)):
                    for element in element_value:
                        element.to_xml(parent=annotation)
                else:
                    element_value.to_xml(parent=annotation)
        return annotation

    def get_text_for_offsets(self, start: int, end: int) -> str:
        """
        Get the text between two offsets. Offsets should not be adjusted to account for the document's start_offset,
        as that is handled within this method.
        Args:
            start: Starting (absolute) offset
            end: Ending (absolute) offset

        Returns:
            The text between the two specified offsets
        """
        return self.elements[RAW_TEXT].raw_text[start - self.start_offset:end - self.start_offset]

    def get_text_for_span(self, span: Span) -> str:
        """Get the text for a given Span contained within this document.
           clao.get_text_for_span(span) and span.get_text_from(clao) are equivalent calls.
        """
        return self.get_text_for_offsets(span.start_offset, span.end_offset)


class IdSpan(IdCLAOElement, Span):
    """A basic Span with an id"""

    def __init__(self, start_offset: int, end_offset: int, element_id: int, span_map=None, **kwargs):
        super().__init__(element_id=element_id, start_offset=start_offset, end_offset=end_offset, span_map=span_map,
                         **kwargs)

    @classmethod
    def from_span(cls, span: Span, elemend_id: int) -> 'IdSpan':
        """Given a Span and an id, create an IdSpan"""
        return IdSpan(span.start_offset, span.end_offset, elemend_id, span.map)

    def __eq__(self, other):
        return super().__eq__(other) and self.element_id == other.element_id


class Embedding(IdCLAOElement):
    """CLAO element representing an embedding vector"""
    element_name = EMBEDDING

    def __init__(self, element_id: int, vector: np.ndarray, **kwargs):
        super(Embedding, self).__init__(element_id=element_id, **kwargs)
        self.vector = vector

    def to_json(self) -> Dict:
        return {**super(Embedding, self).to_json(), VECTOR: list(self.vector)}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        vector = attribs.pop(VECTOR)
        embedding = super(Embedding, self).to_xml(parent, attribs)
        embedding.text = str(vector)
        return embedding

    def __eq__(self, other):
        return super(Embedding, self).__eq__() and all(self.vector == other.vector)

    def __str__(self):
        return f'Embedding({len(self.vector)})'


class EmbeddingContainer(CLAOElementContainer):
    """Add class to any CLAOElement class to create a container for an embedding"""
    def __init__(self, embedding_id: int, clao: TextCLAO, **kwargs):
        self._embedding_id = embedding_id
        super(EmbeddingContainer, self).__init__(clao=clao, **kwargs)

    @property
    def embedding(self) -> Optional[Embedding]:
        if self._embedding_id is not None:
            return self.clao.get_annotations(EMBEDDINGS, self._embedding_id)
        else:
            return None


class Token(IdSpan, EmbeddingContainer):
    """CLAO element representing a token

    Properties:
        embedding: Optional word embedding for this token
    """
    element_name = TOKEN

    def __init__(self, start_offset: int, end_offset: int, element_id: int, clao: TextCLAO, text: str,
                 embedding_id: int = None, span_map=None, **kwargs):
        super(Token, self).__init__(start_offset=start_offset, end_offset=end_offset, element_id=element_id, clao=clao,
                                    embedding_id=embedding_id, span_map=span_map, **kwargs)
        self.text = text

    @classmethod
    def from_id_span(cls, span: IdSpan, text: str, clao: TextCLAO) -> 'Token':
        """Given an IdSpan, create a Token"""
        return Token(span.start_offset, span.end_offset, span.element_id, clao, text, span.map)

    def to_json(self) -> Dict:
        json_dict = {**super(Token, self).to_json(), TEXT: self.text}
        if self._embedding_id is not None:
            json_dict[EMBEDDING_ID] = self._embedding_id
        return json_dict

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        if EMBEDDING_ID in attribs:
            attribs[EMBEDDING_ID] = str(attribs[EMBEDDING_ID])

        text = attribs.pop(TEXT)
        token = super(Token, self).to_xml(parent, attribs)
        token.text = text
        return token

    def __str__(self):
        return f'Token({self.start_offset}, {self.end_offset}, {self.text})'

    def __eq__(self, other):
        return super().__eq__(other) and self._embedding_id == other._embedding_id


class TokenContainer(CLAOElementContainer):
    """Add class to any CLAOElement class to create a container for tokens"""
    def __init__(self, token_id_range: Tuple[int, int], clao: TextCLAO, **kwargs):
        self._token_id_range = token_id_range
        super(TokenContainer, self).__init__(clao=clao, **kwargs)

    @property
    def tokens(self) -> List[Token]:
        if self._token_id_range:
            return self.clao.get_annotations(TOKENS, self._token_id_range)
        else:
            return []


class Heading(Span):
    """CLAO element representing a heading"""
    element_name = HEADING

    def __init__(self, start_offset: int, end_offset: int, text: str):
        super(Heading, self).__init__(start_offset, end_offset)
        self.text = text

    def to_json(self) -> Dict:
        return {**super(Heading, self).to_json(), TEXT: self.text}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        text = attribs.pop(TEXT)
        heading = super(Heading, self).to_xml(parent, attribs)
        heading.text = text
        return heading

    def __str__(self):
        return f'Heading({self.start_offset}, {self.end_offset}, {self.text})'

    def __eq__(self, other):
        return super().__eq__(other) and self.text == other.text


class Entity(IdSpan, TokenContainer):
    """CLAO element representing an entity

    Properties:
        tokens: List of tokens that make up the entity
        entity_type: Type of entity represented (e.g 'ner' or 'fact')
        confidence: percentage of confidence in accuracy of entity
    """
    element_name = ENTITY

    def __init__(self, element_id: int, entity_type: str, confidence: float, token_id_range: Tuple[int, int],
                 clao: TextCLAO, span_map=None):
        self.entity_type = entity_type
        self.confidence = confidence

        start_offset = clao.get_annotations(TOKENS, token_id_range[0]).start_offset
        end_offset = clao.get_annotations(TOKENS, token_id_range[1]-1).end_offset
        super(Entity, self).__init__(start_offset=start_offset, end_offset=end_offset, element_id=element_id, clao=clao,
                                     token_id_range=token_id_range, span_map=span_map)

        self.text = self.get_text_from(clao)

    def to_json(self) -> Dict:
        return {**super(Entity, self).to_json(),
                'token_ids': f"[{self._token_id_range[0]}, {self._token_id_range[1]})",
                'type': self.entity_type,
                'confidence': str(self.confidence),
                TEXT: self.text}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        text = attribs.pop(TEXT)
        entity = super(Entity, self).to_xml(parent, attribs)
        entity.text = text
        return entity

    def __str__(self):
        return f'Entity({self.start_offset}, {self.end_offset}, {self.entity_type}, tokens: {len(self.tokens)})'

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.entity_type == other.entity_type \
               and self.confidence == other.confidence \
               and self._token_id_range == other._token_id_range


class EntityContainer(CLAOElementContainer):
    """Add class to any CLAOElement class to create a container for entities"""
    def __init__(self, entity_id_range: Tuple[int, int], clao: TextCLAO, **kwargs):
        self._entity_id_range = entity_id_range
        super(EntityContainer, self).__init__(clao=clao, **kwargs)

    @property
    def entities(self) -> List[Entity]:
        if self._entity_id_range:
            return self.clao.get_annotations(ENTITIES, self._entity_id_range)
        else:
            return []


class Sentence(IdSpan, TokenContainer, EntityContainer, EmbeddingContainer):
    """CLAO element representing a sentence

    Properties:
        tokens: List of tokens contained in the sentence
        entities: List of entities contained in the sentence
        embedding: Optional word embedding for this token
    """
    element_name = SENTENCE

    def __init__(self, start_offset: int, end_offset: int, element_id: int, clao: TextCLAO,
                 entity_id_range: Optional[Tuple[int, int]] = None, token_id_range: Optional[Tuple[int, int]] = None,
                 embedding_id: int = None, span_map=None, **kwargs):
        super(Sentence, self).__init__(start_offset=start_offset, end_offset=end_offset, element_id=element_id,
                                       clao=clao, entity_id_range=entity_id_range, token_id_range=token_id_range,
                                       embedding_id=embedding_id, span_map=span_map, **kwargs)

    def all_subspans(self) -> List[Span]:
        """Get all Spans contained within this document in a single list

        Returns: a unified list with all the members of this object that are Spans
        """
        all_spans: List[Span] = []
        all_spans.extend(self.tokens)
        all_spans.extend(self.entities)
        return all_spans

    def adjust_offsets(self, delta: int) -> None:
        """Adjust the start and end offsets of this Sentence (and any existing spans therein) by the provided delta

        Args:
            delta: How much to shift the offsets. To reduce the offsets, this should be negative.

        Returns: None
        """
        super().adjust_offsets(delta)
        for span in self.all_subspans():
            span.adjust_offsets(delta)

    @classmethod
    def from_id_span(cls, span: IdSpan, clao: TextCLAO, entity_id_range: Optional[Tuple[int, int]] = None,
                     token_id_range: Optional[Tuple[int, int]] = None, embedding_id: int = None) -> 'Sentence':
        """Given an IdSpan and lists of Tokens, Entities, etc create a new Sentence"""
        return cls(span.start_offset, span.end_offset, span.element_id, clao, entity_id_range, token_id_range,
                   embedding_id, span.map)

    def to_json(self) -> Dict:
        json_dict = {**super(Sentence, self).to_json(),
                     ENTITIES: [e.to_json() for e in self.entities],
                     TOKENS: [t.to_json() for t in self.tokens]}

        if self._embedding_id is not None:
            json_dict[EMBEDDING_ID] = self._embedding_id

        return json_dict

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        if EMBEDDING_ID in attribs:
            attribs[EMBEDDING_ID] = str(attribs[EMBEDDING_ID])
        attribs.pop(ENTITIES)
        attribs.pop(TOKENS)
        sentence = super(Sentence, self).to_xml(parent, attribs)
        for entity in self.entities:
            entity.to_xml(parent=sentence)
        for token in self.tokens:
            token.to_xml(sentence)
        return sentence

    def get_tokens_for_span(self, span: Span, partial=False) -> List[Token]:
        """Get the Tokens in this sentence that are covered by a given Span

        Args:
            span: the Span to retrieve Tokens for
            partial: if True, will return Tokens that are only partially covered by the Span.
                     Otherwise, a Token will only be included if it is fully covered by the Span.
        Returns:
            A list of Tokens covered by the Span
        """
        if span.contains(self):
            return self.tokens
        elif partial:
            return [t for t in self.tokens if t.overlaps(span)]
        else:
            return [t for t in self.tokens if span.contains(t)]

    def get_entity_spans_for_span(self, span, partial=False) -> List[Entity]:
        """Get the Entities in this sentence that are covered by a given Span

        Args:
            span: the Span to retrieve Entities for
            partial: if True, will return Entities that are only partially covered by the Span.
                     Otherwise, a Entity will only be included if it is fully covered by the Span.
        Returns:
            A list of Entities covered by the Span
        """
        if span.contains(self):
            return self.entities
        elif partial:
            return [es for es in self.entities if es.overlaps(span)]
        else:
            return [es for es in self.entities if span.contains(es)]

    def __str__(self):
        return f'Sentence({self.start_offset}, {self.end_offset}, tokens: {len(self.tokens)}, ' \
               f'entities: {len(self.entities)})'

    def __eq__(self, other):
        return super().__eq__(other) \
               and self._token_id_range == other._token_id_range \
               and self._entity_id_range == other._entity_id_range \
               and self._embedding_id == other._embedding_id


class SentenceContainer(CLAOElementContainer):
    """Add class to any CLAOElement class to create a container for sentences"""
    def __init__(self, sentence_id_range: Tuple[int, int], clao: TextCLAO, **kwargs):
        self._sentence_id_range = sentence_id_range
        super(SentenceContainer, self).__init__(clao=clao, **kwargs)

    @property
    def sentences(self) -> List[Sentence]:
        if self._sentence_id_range:
            return self.clao.get_annotations(SENTENCES, self._sentence_id_range)
        else:
            return []


class Paragraph(CLAOElement, SentenceContainer):
    """CLAO element representing a paragraph

    Properties:
        sentences: List of sentences contained in the paragraph
    """
    element_name = PARAGRAPH

    def __init__(self, clao: TextCLAO, sentence_id_range: Optional[Tuple[int, int]] = None, **kwargs):
        super(Paragraph, self).__init__(clao=clao, sentence_id_range=sentence_id_range, **kwargs)

    def to_json(self) -> Dict:
        return {**super(Paragraph, self).to_json(), SENTENCES: [s.to_json() for s in self.sentences]}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        attribs.pop(SENTENCES)
        paragraph = super(Paragraph, self).to_xml(parent, attribs)
        for sentence in self.sentences:
            sentence.to_xml(parent=paragraph)
        return paragraph

    def __str__(self):
        return f'Paragraph(sentences: {len(self.sentences)})'

    def __eq__(self, other):
        return super().__eq__(other) and self._sentence_id_range == other._sentence_id_range


class ParagraphContainer(CLAOElementContainer):
    """Add class to any CLAOElement class to create a container for paragraphs"""
    def __init__(self, paragraph_id_range: Tuple[int, int], clao: TextCLAO, **kwargs):
        self._paragraph_id_range = paragraph_id_range
        super(ParagraphContainer, self).__init__(clao=clao, **kwargs)

    @property
    def paragraphs(self) -> List[Paragraph]:
        if self._paragraph_id_range:
            return self.clao.get_annotations(PARAGRAPHS, self._paragraph_id_range)
        else:
            return []


class Section(Span, ParagraphContainer):
    """CLAO element representing a section

    Properties:
        paragraphs: List of paragraphs contained in the section
        heading: Optional heading object for the section
    """
    element_name = SECTION

    def __init__(self, start_offset: int, end_offset: int, clao: TextCLAO,
                 paragraph_id_range: Optional[Tuple[int, int]] = None, heading_id: Optional[int] = None, **kwargs):
        super(Section, self).__init__(start_offset=start_offset, end_offset=end_offset, clao=clao,
                                      paragraph_id_range=paragraph_id_range, **kwargs)
        self._heading_id = heading_id

    @property
    def heading(self) -> Optional[Heading]:
        if self._heading_id is not None:
            return self.clao.get_annotations(HEADINGS, self._heading_id)
        else:
            return None

    def to_json(self) -> Dict:
        json_dict = {**super(Section, self).to_json(), PARAGRAPHS: [p.to_json() for p in self.paragraphs]}
        if self.heading is not None:
            json_dict[HEADING] = self.heading.to_json()
        return json_dict

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        attribs.pop(PARAGRAPHS)
        if HEADING in attribs:
            attribs.pop(HEADING)
        section = super(Section, self).to_xml(parent, attribs)

        if self.heading is not None:
            self.heading.to_xml(parent=section)
        for paragraph in self.paragraphs:
            paragraph.to_xml(parent=section)
        return section

    def __str__(self):
        return f'Section({self.start_offset}, {self.end_offset}, paragraphs: {len(self.paragraphs)})'

    def __eq__(self, other):
        return super().__eq__(other) \
               and self._paragraph_id_range == other._paragraph_id_range \
               and self._heading_id == other._heading_id


class RawText(Span):
    """CLAO object representing unannotated text"""
    element_name = RAW_TEXT

    def __init__(self, rax_text: str):
        super(RawText, self).__init__(0, len(rax_text))
        self.raw_text = rax_text

    def to_json(self) -> Dict:
        return {**super(RawText, self).to_json(), RAW_TEXT: self.raw_text}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        text = attribs.pop(RAW_TEXT)
        raw_text = super(RawText, self).to_xml(parent, attribs)
        raw_text.text = text
        return raw_text

    def __str__(self):
        return f'RawText({len(self.raw_text)})'

    def __eq__(self, other):
        return super().__eq__(other) and self.raw_text == other.raw_text
