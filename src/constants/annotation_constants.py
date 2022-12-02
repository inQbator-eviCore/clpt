from enum import Enum


META_INFO = 'metadata'
METRIC = 'metric'
ANNOTATION = 'annotation'
ACTUAL_LABEL = 'actual_label'
CLEANED_TEXT = 'cleaned_text'
DESCRIPTION = 'description'
DOCUMENT_NAME = 'doc_name'
EMBEDDING = 'embedding'
EMBEDDING_ID = 'embedding_id'
ENTITIES = 'entities'
ENTITY = 'entity'
ENTITY_GROUP = 'entity_group'
ELEMENT = 'element'
EXPAND_ABBREVIATION = 'expand_abbreviation'
HEADING = 'heading'
ID = 'id'
LEMMA = 'lemma'
LITERAL = 'literal'
PARAGRAPH = 'paragraph'
PARAGRAPHS = 'paragraphs'
POS = 'pos'
PROBABILITY = 'probability'
PREDICTION = 'predicted_label'
RAW_TEXT = 'raw_text'
SECTION = 'section'
SENTENCE = 'sentence'
SENTENCES = 'sentences'
SPAN = 'span'
SPELL_CORRECTED_TOKEN = 'spell_corrected_token'
STEM = 'stem'
TEXT = 'text'
TEXT_ELEMENT = 'text'
TOKEN = 'token'
TOKENS = 'tokens'
VECTOR = 'vector'
EMBEDDING_VECTOR = 'embedding_vector'


class EntityType(Enum):
    """Types of Entities in a TextCLAO"""
    FACT = 'FACT'
    MENTION = 'MENTION'
    NER = 'NER'


# The below example of some abbreviation for expansion is just for illustration purpose
# TODO: We should replace with a more detailed dictionary, such as abbreviation for clinical texts
ABBR_DICT = {
    "Pa": "Pseudomonas aeruginosa",
    "CF": "cystic fibrosis",
    "cf": "cystic fibrosis",
    "cpa": "chronic pa infection",
    "DCTN4": "dynactin 4",
    "pa": "Pseudomonas Aeruginosa",

    "what's": "what is",
    "what're": "what are",
    "who's": "who is",
    "who're": "who are",
    "where's": "where is",
    "where're": "where are",
    "when's": "when is",
    "when're": "when are",
    "how's": "how is",
    "how're": "how are",

    "i'm": "i am",
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "there're": "there are",

    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
    "they've": "they have",
    "who've": "who have",
    "would've": "would have",
    "not've": "not have",

    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "they'll": "they will",

    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "can't": "can not",
    "couldn't": "could not",
    "don't": "do not",
    "didn't": "did not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not"
}
