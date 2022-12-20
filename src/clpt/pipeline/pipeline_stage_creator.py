"""Function to build pipeline stages."""
import logging
from typing import List, Tuple

from omegaconf import DictConfig
from src.clpt.pipeline.stages.analysis.doc_cleaner import ConvertToLowerCase, DoNothingDocCleaner, \
    ExcludePunctuation, RemoveStopWord, ExcludeNumbers
from src.clpt.pipeline.stages.analysis.embeddings import FastTextEmbeddings, WordVecEmbeddings, \
    SentenceEmbeddings, WordEmbeddings
from src.clpt.pipeline.stages.analysis.tf_idf_processor import tfidf_vector_processor
from src.clpt.pipeline.stages.analysis.lemmatization import SpaCyLemma, WordnetLemma
from src.clpt.pipeline.stages.analysis.pos_tagger import SimplePOSTagger
from src.clpt.pipeline.stages.analysis.sentence_breaking import RegexSentenceBreaking, SentenceBreaking
from src.clpt.pipeline.stages.analysis.spacy_processing import SpaCyProcessing
from src.clpt.pipeline.stages.analysis.spell_correct import SpellCorrectLevenshtein
from src.clpt.pipeline.stages.analysis.stemming import PorterStemming
from src.clpt.pipeline.stages.analysis.cluster import Cluster
from src.clpt.pipeline.stages.analysis.transcribe import Transcribe
from src.clpt.pipeline.stages.analysis.tokenization import RegexTokenization, WhitespaceRegexTokenization
from src.clpt.pipeline.stages.classification.abbreviation_expansion import AbbreviationExpandWithDict
from src.clpt.pipeline.stages.classification.MLClassifier import ML_Model
# SpacyAbbreviationExpand
from src.clpt.pipeline.stages.classification.entities import CoreferenceResolution, FactExtraction, GroupEntities, \
    MentionDetection, RelationExtraction
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.constants import CONFIG_STAGE_KEY

logger = logging.getLogger(__name__)
'''
ALL_KNOWN_STAGES = [AbbreviationExpandWithDict, SpacyAbbreviationExpand, ConvertToLowerCase,
                    CoreferenceResolution, DoNothingDocCleaner, ExcludePunctuation,ExcludeNumbers,
                    FactExtraction, FastTextEmbeddings,WordVecEmbeddings,
                    GroupEntities, MentionDetection, PorterStemming, RegexSentenceBreaking, RegexTokenization,
                    RelationExtraction, RemoveStopWord, SentenceBreaking, SentenceEmbeddings, SimplePOSTagger,
                    SpaCyLemma, SpaCyProcessing, SpellCorrectLevenshtein, WhitespaceRegexTokenization, WordEmbeddings,
                    WordnetLemma, ML_Model,tfidf_vector_processor]
STAGE_TYPES = {s.__name__: s for s in ALL_KNOWN_STAGES}
'''
ALL_KNOWN_STAGES = [AbbreviationExpandWithDict, ConvertToLowerCase,
                    CoreferenceResolution, DoNothingDocCleaner, ExcludePunctuation, ExcludeNumbers, FactExtraction,
                    FastTextEmbeddings, WordVecEmbeddings, WordnetLemma, ML_Model, tfidf_vector_processor,
                    GroupEntities, MentionDetection, PorterStemming, RegexSentenceBreaking, RegexTokenization,
                    RelationExtraction, RemoveStopWord, SentenceBreaking, SentenceEmbeddings, SimplePOSTagger,
                    SpaCyLemma, SpaCyProcessing, SpellCorrectLevenshtein, WhitespaceRegexTokenization, WordEmbeddings,
                    Transcribe, Cluster]
STAGE_TYPES = {s.__name__: s for s in ALL_KNOWN_STAGES}


def build_pipeline_stages(cfg: DictConfig) -> Tuple[List[PipelineStage], List[PipelineStage]]:
    """Build pipeline stages.

    Takes as input some YAML config and generates a number of stage objects from it. Raise error, such as when required
    arguments are missing or if an unexpected argument is passed through.

    Args:
        cfg: DictConfig laoded for pipeline.

    Return:
        stages (list): A list of class objects for pipeline stages.
    """
    single_clao_stages = []
    all_claos_stages = []
    for stage_cfg in cfg.analysis.pipeline_stages:
        stage_dict = dict(stage_cfg)
        stage_type = stage_dict.pop(CONFIG_STAGE_KEY)

        try:
            stage_cls = STAGE_TYPES[stage_type]
            logger.info(f"Loading {stage_type}...")
            stage = stage_cls.from_config(**stage_dict)
            if stage.single_clao:
                single_clao_stages.append(stage)
            else:
                all_claos_stages.append(stage)
        except TypeError as e:
            logger.error(f"Error encountered when loading {stage_type}: arguments '{stage_type}' does not match "
                         f"required arguments for {stage_type}")
            raise e
        except FileNotFoundError as e:
            logger.error(f"Error encountered when loading {stage_type}: File in arguments for {stage_type} "
                         f"not found: '{stage_cfg}'")
            raise e
        except Exception as e:
            logger.error(f"{type(e).__name__} encountered when loading {stage_type}: arguments '{stage_cfg}'.")
            raise e

    if "pipeline_stages" in cfg.classification:
        for stage_cfg in cfg.classification.pipeline_stages:
            stage_dict = dict(stage_cfg)
            stage_type = stage_dict.pop(CONFIG_STAGE_KEY)
            try:
                stage_cls = STAGE_TYPES[stage_type]
                logger.info(f"Loading {stage_type}...")
                stage = stage_cls.from_config(**stage_dict)
                if stage.single_clao:
                    single_clao_stages.append(stage)
                else:
                    all_claos_stages.append(stage)
            except TypeError as e:
                logger.error(f"Error encountered when loading {stage_type}: arguments '{stage_type}' does not match "
                             f"required arguments for {stage_type}")
                raise e
            except FileNotFoundError as e:
                logger.error(f"Error encountered when loading {stage_type}: File in arguments for {stage_type} "
                             f"not found: '{stage_cfg}'")
                raise e
            except Exception as e:
                logger.error(f"{type(e).__name__} encountered when loading {stage_type}: arguments '{stage_cfg}'.")
                raise e
    return single_clao_stages, all_claos_stages
