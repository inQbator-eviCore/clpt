"""Function to build pipeline stages."""
import logging
from typing import List, Tuple

from omegaconf import DictConfig

from src.clpt.pipeline.stages.doc_cleaner import ConvertToLowerCase, DoNothingDocCleaner, ExcludePunctuation, \
    RemoveStopWord
from src.clpt.pipeline.stages.embeddings import FastTextEmbeddings, SentenceEmbeddings, WordEmbeddings
from src.clpt.pipeline.stages.entities import GroupEntities, MentionDetection
from src.clpt.pipeline.stages.lemmatization import SpaCyLemma, WordnetLemma
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.clpt.pipeline.stages.pos_tagger import SimplePOSTagger
from src.clpt.pipeline.stages.sentence_breaking import RegexSentenceBreaking, SentenceBreaking
from src.clpt.pipeline.stages.spacy_processing import SpaCyProcessing
from src.clpt.pipeline.stages.spell_correct import SpellCorrectLevenshtein
from src.clpt.pipeline.stages.stemming import PorterStemming
from src.clpt.pipeline.stages.tokenization import RegexTokenization, WhitespaceRegexTokenization
from src.constants.constants import CONFIG_STAGE_KEY

logger = logging.getLogger(__name__)

ALL_KNOWN_STAGES = [ConvertToLowerCase, DoNothingDocCleaner, ExcludePunctuation, FastTextEmbeddings, GroupEntities,
                    MentionDetection, PorterStemming, RegexSentenceBreaking, RegexTokenization, RemoveStopWord,
                    SentenceBreaking, SentenceEmbeddings, SimplePOSTagger, SpaCyLemma, SpaCyProcessing,
                    SpellCorrectLevenshtein, WhitespaceRegexTokenization, WordEmbeddings, WordnetLemma]
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

    return single_clao_stages, all_claos_stages
