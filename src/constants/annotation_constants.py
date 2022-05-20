# Span
from enum import Enum

START_OFFSET = 'startOffset'
END_OFFSET = 'endOffset'
MAP = 'map'

ANNOTATION = 'annotation'
EMBEDDING = 'embedding'
EMBEDDING_ID = 'embedding_id'
EMBEDDINGS = 'embeddings'
ENTITIES = 'entities'
ENTITY = 'entity'
ENTITY_GROUP = 'entity_group'
ENTITY_GROUPS = 'entity_groups'
ELEMENT = 'element'
HEADING = 'heading'
HEADINGS = 'headings'
KEY = 'key'
LEMMA = 'lemma'
LITERAL = 'literal'
PATTERN = 'pattern'
PARAGRAPH = 'paragraph'
PARAGRAPHS = 'paragraphs'
POS = 'pos'
RAW_TEXT = 'raw_text'
SECTION = 'section'
SENTENCE = 'sentence'
SENTENCES = 'sentences'
SPAN = 'span'
SPELL_CORRECTED_TOKEN = 'spell_corrected_token'
STEM = 'stem'
TEXT = 'text'
TOKEN = 'token'
TOKENS = 'tokens'
VECTOR = 'vector'

INDEX = 'index'
ID = 'id'
CLEANED_TEXT = 'cleaned_text'


class EntityType(Enum):
    """Types of Entities in a TextCLAO"""
    FACT = 'FACT'
    MENTION = 'MENTION'
    NER = 'NER'
