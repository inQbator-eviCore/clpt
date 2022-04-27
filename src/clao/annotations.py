"""Initial thoughts / rough draft of some internal text annotation objects, in line with XML schema illustrated in 2022
NaaCL paper"""

from typing import Dict, Iterable, List, Optional

from lxml import etree

from src.constants.annotation_constants import ANNOTATION, ELEMENT, ENTITIES, ENTITY, HEADING, ID, PARAGRAPH, \
    PARAGRAPHS, RAW_TEXT, SECTION, SENTENCE, SENTENCES, SPAN, TEXT, TOKEN, TOKENS


class CLAOElement:
    element_name = ELEMENT

    def to_json(self) -> Dict:
        return {}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs is None:
            attribs = self.to_json()
        if parent is None:
            return etree.Element(self.element_name, **attribs)
        else:
            return etree.SubElement(parent, self.element_name, **attribs)


class Span(CLAOElement):
    element_name = SPAN

    def __init__(self, start_offset: int, end_offset: int, span_map=None):
        """add docstring here"""
        super(Span, self).__init__()
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.map = {} if span_map is None else span_map

    def adjust_offsets(self, delta: int) -> None:
        """
        Adjust the start and end offsets of this span by the provided delta
        Args:
            delta: How much to shift the offsets. To reduce the offsets, this should be negative.

        Returns: None
        """
        self.start_offset += delta
        self.end_offset += delta

    def to_json(self) -> Dict:
        return {'start': str(self.start_offset),
                'end': str(self.end_offset),
                **super(Span, self).to_json()}

    def get_text_from(self, annotation: 'Annotations') -> str:
        """
        """
        return annotation.get_text_for_span(self)

    def overlaps(self, other):
        """
        Args:
            other: a Span
        Returns:
            Whether or not these two spans overlap at all
        """
        return self.start_offset < other.end_offset and self.end_offset > other.start_offset

    def contains(self, other):
        """
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


class IdSpan(Span):
    """A basic Span with an index.

    Attributes:
        index: The index of this Span (starts at 0).
    """
    ID_FIELD = ID

    def __init__(self, start_offset: int, end_offset: int, index: int, span_map=None):
        super().__init__(start_offset, end_offset, span_map)
        self.index = index

    @classmethod
    def from_span(cls, span: Span, index: int) -> 'IdSpan':
        """Given a Span and an index, create an IdSpan"""
        return IdSpan(span.start_offset, span.end_offset, index, span.map)

    def to_json(self):
        """Convert this IdSpan to a JSON map (invoked by subclasses)
        Returns:
            dictionary that serves as a JSON map for this object
        """
        return {self.ID_FIELD: self.index, **super().to_json()}

    def __eq__(self, other):
        return super().__eq__(other) and self.index == other.index


class Token(IdSpan):
    element_name = TOKEN

    def __init__(self, start_offset: int, end_offset: int, index: int, text: str, span_map=None):
        """add docstring here"""
        super(Token, self).__init__(start_offset, end_offset, index, span_map)
        self.text = text

    @classmethod
    def from_id_span(cls, span: IdSpan, text: str) -> 'Token':
        """Given an IdSpan, create a Token"""
        return Token(span.start_offset, span.end_offset, span.index, text, span.map)

    def to_json(self) -> Dict:
        return {TEXT: self.text, **super(Token, self).to_json()}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        text = attribs.pop(TEXT)
        token = super(Token, self).to_xml(parent, attribs)
        token.text = text
        return token

    def __str__(self):
        return f'Token({self.start_offset}, {self.end_offset})'

    def __eq__(self, other):
        return super().__eq__(other)


class Heading(Span):
    element_name = HEADING

    def __init__(self, start_offset: int, end_offset: int, text: str):
        """add docstring here"""
        super(Heading, self).__init__(start_offset, end_offset)
        self.text = text

    def to_json(self) -> Dict:
        return {TEXT: self.text, **super(Heading, self).to_json()}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        text = attribs.pop(TEXT)
        heading = super(Heading, self).to_xml(parent, attribs)
        heading.text = text
        return heading


class Entity(Span):
    element_name = ENTITY

    def __init__(self, tokens: Iterable[Token], entity_type: str, confidence: float, text: str):
        """add docstring here"""
        super(Entity, self).__init__()
        self.tokens = tokens
        self.entity_type = entity_type
        self.confidence = confidence
        self.text = text

    def to_json(self) -> Dict:
        token_ids = [token.id for token in self.tokens]
        return {'tokens': str(token_ids),
                'type': self.entity_type,
                'confidence': str(self.confidence),
                TEXT: self.text,
                **super(Entity, self).to_json()}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        text = attribs.pop(TEXT)
        entity = super(Entity, self).to_xml(parent, attribs)
        entity.text = text
        return entity


class Sentence(IdSpan):
    element_name = SENTENCE

    def __init__(self, start_offset: int, end_offset: int, index: int, entities: Iterable[Entity] = (),
                 tokens: Iterable[Token] = (), span_map=None):
        """add docstring here"""
        super(Sentence, self).__init__(start_offset, end_offset, index, span_map)
        self.entities = list(entities)
        self.tokens = list(tokens)

    def all_subspans(self) -> List[Span]:
        """
        Get all spans contained within this document in a single list
        Returns: a unified list with all the members of this object that are Spans
        """
        all_spans: List[Span] = []
        all_spans.extend(self.tokens)
        all_spans.extend(self.entities)
        return all_spans

    def adjust_offsets(self, delta: int) -> None:
        """
        Adjust the start and end offsets of this Sentence (and any existing spans therein) by the provided delta
        Args:
            delta: How much to shift the offsets. To reduce the offsets, this should be negative.

        Returns: None
        """
        super().adjust_offsets(delta)
        for span in self.all_subspans():
            span.adjust_offsets(delta)

    @classmethod
    def from_indexed_span(cls, span: IdSpan, entities: List[Entity], tokens: List[Token], ) -> 'Sentence':
        """Given an IdSpan and lists of Tokens, Entities create a new Sentence"""
        return cls(span.start_offset, span.end_offset, span.index, entities, tokens, span.map)

    def to_json(self) -> Dict:
        return {ENTITIES: [e.to_json() for e in self.entities],
                TOKENS: [t.to_json() for t in self.tokens],
                **super(Sentence, self).to_json()}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        attribs.pop(ENTITIES)
        attribs.pop(TOKENS)
        sentence = super(Sentence, self).to_xml(parent, attribs)
        for entity in self.entities:
            entity.to_xml(parent=sentence)
        for token in self.tokens:
            token.to_xml(sentence)
        return sentence

    def get_tokens_for_span(self, span: Span, partial=False) -> List[Token]:
        """
        Get the Tokens in this sentence that are covered by a given Span
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

    def get_entity_spans_for_span(self, span, partial=False) -> Iterable[Entity]:
        """
        Get the Entities in this sentence that are covered by a given Span
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
        return super().__eq__(other) and self.tokens == other.tokens and \
               self.entities == other.entities


class Paragraph(CLAOElement):
    element_name = PARAGRAPH

    def __init__(self, sentences: Iterable[Sentence]):
        """add docstring here"""
        super(Paragraph, self).__init__()
        self.sentences = sentences

    def to_json(self) -> Dict:
        return {SENTENCES: [s.to_json() for s in self.sentences],
                **super(Paragraph, self).to_json()}

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


class Section(Span):
    element_name = SECTION

    def __init__(self, start_offset: int, end_offset: int, paragraphs: Iterable[Paragraph],
                 heading: Optional[Heading] = None):
        """add docstring here"""
        super(Section, self).__init__(start_offset, end_offset)
        self.paragraphs = paragraphs
        self.heading = heading

    def to_json(self) -> Dict:
        json_dict = {PARAGRAPHS: [p.to_json() for p in self.paragraphs],
                     **super(Section, self).to_json()}
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


class Annotations(Span):
    element_name = ANNOTATION

    def __init__(self, raw_text: str):
        """add docstring here"""
        super(Annotations, self).__init__(0, len(raw_text))
        self.elements = {RawText.element_name: RawText(raw_text)}

    def to_json(self) -> Dict:
        json_dict = super(Annotations, self).to_json()
        for element_type, element_value in self.elements.items():
            if isinstance(element_value, list):
                json_dict[element_type] = [e.to_json() for e in element_value]
            else:
                json_dict.update(element_value.to_json())
        return json_dict

    def to_xml(self) -> etree.Element:
        annotation = super(Annotations, self).to_xml(parent=None, attribs={})
        for element_type, element_value in self.elements.items():
            if isinstance(element_value, list):
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
        """Get the text for a given span contained within this document.
           annotations.get_text_for_span(span) and span.get_text_from(annotations) are equivalent calls."""
        return self.get_text_for_offsets(span.start_offset, span.end_offset)


class RawText(Span):
    element_name = RAW_TEXT

    def __init__(self, rax_text: str):
        """add docstring here"""
        super(RawText, self).__init__(0, len(rax_text))
        self.raw_text = rax_text

    def to_json(self) -> Dict:
        return {RAW_TEXT: self.raw_text,
                **super(RawText, self).to_json()}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        text = attribs.pop(RAW_TEXT)
        raw_text = super(RawText, self).to_xml(parent, attribs)
        raw_text.text = text
        return raw_text
