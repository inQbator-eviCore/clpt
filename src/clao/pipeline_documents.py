from src.clao.text_clao import Span


# TODO: Adapt these into equivalent objects in clao.annotations


class EntitySpan(Span):
    pass
#     """Span representing an identified "entity" within the text.
#
#     The meaning of "entity" is left intentionally open-ended. Currently used to house dictionary-lookup-discovered
#     named entities (aka "facts") and model-discovered named entities.
#
#     Attributes:
#         start_token_index: The index, within a sentence, of the first token included in this span (inclusive)
#         end_token_index: The index, within a sentence, of the last token included in this span (inclusive)
#         entities: A list of Entity objects represented by this span
#     """
#     def __init__(self, start_offset: int, end_offset: int, start_token_index: int, end_token_index: int,
#                  entities: List[Entity] = None, span_map=None):
#         super().__init__(start_offset, end_offset, span_map)
#         self.start_token_index = start_token_index
#         self.end_token_index = end_token_index
#         self.entities = entities if entities is not None else []
#
#     @classmethod
#     def from_span(cls, span: Span, start_token_index, end_token_index, entities: List[Entity]) -> 'EntitySpan':
#         """Given a Span and a list of Entities, create a new EntitySpan"""
#         return cls(span.start_offset, span.end_offset, start_token_index, end_token_index, entities, span.map)
#
#     @classmethod
#     def from_json(cls, j) -> 'EntitySpan':
#         """Parse an EntitySpan from JSON
#
#         Args:
#             j: a JSON map
#         """
#         span = Span.from_json(j)
#
#         try:
#             if FACTS in j:
#                 entities = [Fact.from_json(f) for f in j[FACTS]]
#             else:
#                 entities = [Entity.from_json(e) for e in j[NAMED_ENTITIES]]
#
#             return cls.from_span(span, j[START_TOKEN_INDEX], j[END_TOKEN_INDEX], entities)
#         # The 'namedEntities' check will throw a KeyError if we're looking at old-schema namedEntitySpans
#         except KeyError:
#             return cls.from_legacy_json(j)
#
#     @classmethod
#     def from_legacy_json(cls, j) -> 'EntitySpan':
#         """Parse an EntitySpan from JSON using the legacy schema of handling named entities as a special case
#
#         Args:
#             j: a JSON map
#         """
#         span = Span.from_json(j)
#
#         if FACTS in j:
#             entities = [Fact.from_json(f) for f in j[FACTS]]
#         else:
#             j[ASSERTION_STATUS] = j[MAP][ASSERTION_STATUS]
#             j[VALUE] = j[MAP][VALUE]
#             if TIME in j[MAP]:
#                 j[TIME] = j[MAP][TIME]
#             entities = [Entity.from_json(j)]
#
#         return cls.from_span(span, j[START_TOKEN_INDEX], j[END_TOKEN_INDEX], entities)
#
#     def to_json(self):
#         """Convert this FactSpan to a JSON map.
#
#         This function accounts for the possibility that an EntitySpan could contain a mix of Fact object and
#         other Entity objects. This currently does not happen anywhere, but is supported here just in case.
#
#         Returns:
#             dictionary that serves as a JSON map for this object
#         """
#         common = {START_TOKEN_INDEX: self.start_token_index, END_TOKEN_INDEX: self.end_token_index}
#         facts = []
#         named_entities = []
#         for e in self.entities:
#             (facts if isinstance(e, Fact) else named_entities).append(e)
#         fact_span_json = {}
#         ne_span_json = {}
#         if len(facts) > 0:
#             fact_span_json = {FACTS: [f.to_json() for f in facts], **super().to_json(), **common}
#         if len(named_entities) > 0:
#             ne_span_json = {NAMED_ENTITIES: [e.to_json() for e in named_entities], **super().to_json(), **common}
#         return fact_span_json, ne_span_json
#
#     # TODO: Remove this legacy writer once we're sure we won't use it anymore
#     def to_legacy_json(self):
#         """Convert this FactSpan to a JSON map using the legacy schema of treating named entities as a special case.
#
#         This function accounts for the possibility that an EntitySpan could contain a mix of Fact object and
#         other Entity objects. This currently does not happen anywhere, but is supported here just in case.
#
#         Returns:
#             dictionary that serves as a JSON map for this object
#         """
#         common = {START_TOKEN_INDEX: self.start_token_index, END_TOKEN_INDEX: self.end_token_index}
#         facts = []
#         other = []
#         for e in self.entities:
#             (facts if isinstance(e, Fact) else other).append(e)
#         fact_span_json = {}
#         entity_span_jsons = []
#         if len(facts) > 0:
#             fact_span_json = {FACTS: [f.to_json() for f in facts], **super().to_json(), **common}
#         for e in other:
#             self.map[ASSERTION_STATUS] = e.assertion_status.name
#             self.map[VALUE] = e.value.to_json()
#             self.map[TIME] = e.time.to_json()
#             entity_span_jsons.append({NER_LABEL: e.label, **super().to_json(), **common})
#         return fact_span_json, entity_span_jsons
#
#     def __str__(self):
#         return f'EntitySpan({self.start_offset}, {self.end_offset}, {self.start_token_index}, ' \
#                f'{self.end_token_index}, {[(e.label, e.assertion_status) for e in self.entities]})'
#
#     def __eq__(self, other):
#         return super().__eq__(other) and self.start_token_index == other.start_token_index and \
#                self.end_token_index == other.end_token_index and self.entities == other.entities


# class PipelineDocument(Span):
#     """Represents a single, individual free-text document within the processing pipeline.
#
#     Attributes:
#         text: The text of this document. len(text) should = end_offset - start_offset.
#         doc_id: The ID of the document
#         title: The document title
#         doc_type: The document type
#         date: The datetime for this document
#         metadata: A map of metadata field name -> value
#         markup_tag_spans: Information about markup tags that have been removed from the text
#         dates: DateSpan objects that represent dates found in the text
#         pages: Page spans representing the pages of this document
#         regions: Region objects representing the different sections identified in this document
#     """
#     def __init__(self, start_offset: int, end_offset: int, text: str, doc_id='', title='', doc_type='',
#                  date=None, metadata: dict = None, markup_tag_spans: List[MarkupTagSpan] = (),
#                  dates: List[DateSpan] = (), pages: List[Page] = (), regions: List[Region] = (),
#                  annotations: Dict[str, List['AnnotationSpan']] = None, span_map=None):
#         super().__init__(start_offset, end_offset, span_map)
#         self.text = text
#         self.id = doc_id
#         self.title = title
#         self.type = doc_type
#         self.date = (datetime.now() if date is None else date).replace(microsecond=0)
#         self.metadata = metadata if metadata is not None else {}
#         self.markup_tag_spans = list(markup_tag_spans)
#         self.dates = list(dates)
#         self.pages = list(pages)
#         self.regions = list(regions)
#         self.annotations = annotations if annotations is not None else {}
#
#     def all_subspans(self) -> List[Span]:
#         """
#         Get all spans contained within this document in a single list
#         Returns: a unified list with all the members of this object that are Spans
#         """
#         all_spans: List[Span] = []
#         all_spans.extend(self.markup_tag_spans)
#         all_spans.extend(self.dates)
#         all_spans.extend(self.pages)
#         all_spans.extend(self.regions)
#         for span_list in self.annotations.values():
#             all_spans.extend(span_list)
#         return all_spans
#
#     def adjust_offsets(self, delta: int) -> None:
#         """
#         Adjust the start and end offsets of this document (and any existing spans therein) by the provided delta
#         Args:
#             delta: How much to shift the offsets. To reduce the offsets, this should be negative.
#
#         Returns: None
#         """
#         super().adjust_offsets(delta)
#         for span in self.all_subspans():
#             span.adjust_offsets(delta)
#
#
#     @classmethod
#     def from_text(cls, text: str, start_offset: int = 0) -> 'PipelineDocument':
#         """Creates a PipelineDocument object from text (useful for testing)"""
#         return PipelineDocument(start_offset, start_offset + len(text), text)
#
#     @classmethod
#     def from_span(cls, span, text, doc_id, title, doc_type, date, metadata=None, markup_tag_spans=(), dates=(),
#                   pages=(), regions=(), annotations=None) -> 'PipelineDocument':
#         """Given a Span, document text, id, title, type, date and metadata, and lists of MarkupTagSpans, DateSpans,
#            Pages, and Regions, create a new PipelineDocument"""
#         return PipelineDocument(span.start_offset, span.end_offset, text, doc_id, title, doc_type, date, metadata,
#                                 markup_tag_spans, dates, pages, regions, annotations, span.map)
#
#     @classmethod
#     def from_json(cls, json_doc) -> 'PipelineDocument':
#         """Parse a PipelineDocument from JSON
#         Args:
#             json_doc: a JSON map
#         """
#         span = Span.from_json(json_doc)
#
#         doc_date = datetime.fromtimestamp(json_doc[DATE] / 1000) if DATE in json_doc else datetime.now()
#
#         markup_tag_spans = [MarkupTagSpan.from_json(j)
#                             for j in json_doc[MARKUP_TAGS]
#                             ] if MARKUP_TAGS in json_doc else []
#         dates = [DateSpan.from_json(j) for j in json_doc[DATES]] if DATES in json_doc else []
#         pages = [Page.from_json(j) for j in json_doc[PAGES]] if PAGES in json_doc else []
#         regions = [Region.from_json(j) for j in json_doc[REGIONS]]
#         annotations = {id_str: [AnnotationSpan.from_json(j) for j in spans] for id_str, spans in
#                        json_doc[ANNOTATIONS].items()} if ANNOTATIONS in json_doc else {}
#
#         return PipelineDocument.from_span(span, json_doc[TEXT], json_doc[ID],
#                                           json_doc[TITLE] if TITLE in json_doc else '',
#                                           json_doc[TYPE] if TYPE in json_doc else '',
#                                           doc_date, json_doc[METADATA] if METADATA in json_doc else {},
#                                           markup_tag_spans, dates, pages, regions, annotations)
#
#     def to_json(self):
#         """Convert this PipelineDocument to a JSON map
#         Returns:
#             dictionary that serves as a JSON map for this object
#         """
#         return {ID: self.id, TEXT: self.text, TITLE: self.title, TYPE: self.type, METADATA: self.metadata,
#                 DATE: int(self.date.timestamp() * 1000),
#                 MARKUP_TAGS: [m.to_json() for m in self.markup_tag_spans],
#                 DATES: [d.to_json() for d in self.dates],
#                 PAGES: [p.to_json() for p in self.pages],
#                 REGIONS: [r.to_json() for r in self.regions],
#                 ANNOTATIONS: {id_str: [span.to_json() for span in spans]
#                                        for id_str, spans in self.annotations.items()},
#                 **super().to_json()}
#
#     # TODO: If we begin having documents where pagebreaks are represented differently,
#     # TODO: we will need to update this logic
#     def reconstruct_original_text(self) -> str:
#         page_texts = []
#         for page in self.pages:
#             page_markup_spans = sorted([m for m in self.markup_tag_spans if page.contains(m)],
#                                        key=lambda x: x.start_offset)
#             page_start = page.start_offset
#             page_text = page.get_text_from(self)
#             chars_added = 0
#
#             for markup_span in page_markup_spans:
#                 tag_start = chars_added + markup_span.start_offset - page_start
#                 if markup_span.tag == DEGRADED:
#                     # case for removed degraded text entirely
#                     if markup_span.start_offset == markup_span.end_offset:
#                         to_add = DEGRADED_START + markup_span.map[TEXT] + DEGRADED_END
#                         page_text = page_text[:tag_start] + to_add + page_text[tag_start:]
#                         chars_added += len(to_add)
#                     else:
#                         tag_end = chars_added + markup_span.end_offset - page_start
#                         page_text = page_text[:tag_start] + DEGRADED_START + page_text[tag_start:tag_end] + \
#                                     DEGRADED_END + page_text[tag_end:]
#                         chars_added += len(DEGRADED_START + DEGRADED_END)
#                 else:
#                     tag_text = '<' + markup_span.tag + '>'
#                     page_text = page_text[:tag_start] + tag_text + page_text[tag_start:]
#                     chars_added += len(tag_text)
#
#             if REMOVED_HEADERS in page.map:
#                 page_text = page.map[REMOVED_HEADERS] + page_text
#                 chars_added += len(page.map[REMOVED_HEADERS])
#
#             if REMOVED_FOOTERS in page.map:
#                 page_text += page.map[REMOVED_FOOTERS]
#                 chars_added += len(page.map[REMOVED_FOOTERS])
#
#             page_text += PAGEBREAK + '\n'
#             page_texts.append(page_text)
#
#         return ''.join(page_texts)
#
#     def sentences(self) -> List[Sentence]:
#         """Get all Sentences contained within this PipelineDocument"""
#         return [s for sents in (r.sentences for r in self.regions) for s in sents]
#
#     def entity_spans(self, incl_facts=True, incl_other_ne=True) -> List[EntitySpan]:
#         """Get all EntitySpans contained within this PipelineDocument
#         Args:
#             incl_facts: whether or not to include entity spans in sentence.fact_spans
#             incl_other_ne: whether or not to include entity spans in sentence.named_entity_spans
#         """
#         return self.get_entity_spans_for_span(span=self, incl_facts=incl_facts, incl_other_ne=incl_other_ne)
#
#     def get_text_for_offsets(self, start: int, end: int) -> str:
#         """
#         Get the text between two offsets. Offsets should not be adjusted to account for the document's start_offset,
#         as that is handled within this method.
#         Args:
#             start: Starting (absolute) offset
#             end: Ending (absolute) offset
#
#         Returns:
#             The text between the two specified offsets
#         """
#         return self.text[start - self.start_offset:end - self.start_offset]
#
#     # Use the following accessors to get text/tokens/facts/etc from a span belonging to this document
#     # e.g. for page in doc.pages: page_text = doc.get_text_for_span(page)
#     # e.g. for region in doc.regions: region_fact_spans = doc.get_fact_spans_for_span(region)
#     def get_text_for_span(self, span: Span) -> str:
#         """Get the text for a given span contained within this document.
#            doc.get_text_for_span(span) and span.get_text_from(doc) are equivalent calls."""
#         return self.get_text_for_offsets(span.start_offset, span.end_offset)
#
#     def get_sentences_for_span(self, span: Span, partial=False) -> List[Sentence]:
#         """
#         Get the Sentences in this PipelineDocument that are covered by a given Span
#         Args:
#             span: the Span to retrieve Sentences for
#             partial: if True, will return Sentences that are only partially covered by the Span.
#                      Otherwise, a Sentence will only be included if it is fully covered by the Span.
#         Returns:
#             A list of Sentences covered by the Span
#         """
#         if self == span:
#             return self.sentences()
#         elif partial:
#             return [s for s in self.sentences() if s.overlaps(span)]
#         else:
#             return [s for s in self.sentences() if span.contains(s)]
#
#     def get_tokens_for_span(self, span: Span, partial=False) -> List[Token]:
#         """
#         Get the Tokens in this PipelineDocument that are covered by a given Span
#         Args:
#             span: the Span to retrieve Tokens for
#             partial: if True, will return Tokens that are only partially covered by the Span.
#                      Otherwise, a Token will only be included if it is fully covered by the Span.
#         Returns:
#             A list of Tokens covered by the Span
#         """
#         sentences = self.get_sentences_for_span(span, partial=True)
#         return [t for tokens in (s.get_tokens_for_span(span, partial) for s in sentences) for t in tokens]
#
#     def get_entity_spans_for_span(self, span: Span, partial=False, incl_facts=True, incl_other_ne=True
#                                   ) -> List[EntitySpan]:
#         """
#         Get the EntitySpans in this PipelineDocument that are covered by a given Span
#         Args:
#             span: the Span to retrieve EntitySpans for
#             partial: if True, will return EntitySpans that are only partially covered by the Span.
#                      Otherwise, a EntitySpan will only be included if it is fully covered by the Span.
#             incl_facts: whether or not to include entity spans in sentence.fact_spans
#             incl_other_ne: whether or not to include entity spans in sentence.named_entity_spans
#         Returns:
#             A list of EntitySpans covered by the Span
#         """
#         sentences = self.get_sentences_for_span(span, partial=True)
#         spans_per_sentence = [s.get_entity_spans_for_span(span=span, partial=partial, incl_facts=incl_facts,
#                                                           incl_other_ne=incl_other_ne) for s in sentences]
#         return [es for entity_spans in spans_per_sentence for es in entity_spans]
#
#     def set_properties_from(self, other: 'PipelineDocument') -> None:
#         """
#         Use a given Pipeline document to set properties and metadata on this one
#         Args:
#             other: the PipelineDocument to take properties/metadata from
#         Returns: None
#         """
#         self.id = other.id
#         self.title = other.title
#         self.type = other.type
#         self.date = other.date
#         self.merge_metadata(other.metadata)
#
#     def merge_metadata(self, old_metadata: dict) -> None:
#         """
#         Merge self.metadata with the provided metadata, giving precedence to self.metadata when keys overlap
#         Args:
#             old_metadata: The metadata to merge in
#         Returns: None
#         """
#         m = old_metadata.copy()
#         m.update(self.metadata)
#         self.metadata = m
#
#     def __str__(self):
#         return f'PipelineDocument({self.id}, {self.date.strftime("%m/%d/%Y %H:%M:%S")}, {self.start_offset}, ' \
#                f'{self.end_offset}, pages: {len(self.pages)}, regions: {len(self.regions)}, ' \
#                f'sentences: {len(self.sentences())}, metadata: {self.metadata})'
#
#     def __eq__(self, other):
#         return super().__eq__(other) and self.text == other.text and self.id == other.id \
#                and self.title == other.title and self.type == other.type and self.date == other.date \
#                and self.metadata == other.metadata and self.markup_tag_spans == other.markup_tag_spans \
#                and self.dates == other.dates and self.pages == other.pages and self.regions == other.regions
