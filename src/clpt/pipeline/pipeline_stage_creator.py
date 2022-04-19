"""Function to build pipeline stages."""
import logging
from typing import List
import yaml

from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.clpt.pipeline.constants import CONFIG_STAGE_KEY
from src.clpt.pipeline.stages.doc_cleaner import DocumentCleaner
from src.clpt.pipeline.stages.sentence_breaking import SentenceBreaking
from src.clpt.pipeline.stages.tokenization import RegexTokenization


logger = logging.getLogger(__name__)

ALL_KNOWN_STAGES = [DocumentCleaner, SentenceBreaking, RegexTokenization]
STAGE_TYPES = {s.__name__: s for s in ALL_KNOWN_STAGES}


def build_pipeline_stages(config_file_path) -> List[PipelineStage]:
    """Build pipeline stages.

    Takes as input some YAML config and generates a number of stage objects from it. Raise error, such as when required
    arguments are missing or if an unexpected argument is passed through.

    Args:
        config_file_path (str): An pathway of the YAML config file.

    Return:
        stages (list): A list of class objects for pipeline stages.
    """

    stages = []
    with open(config_file_path) as data_file:
        yaml_configs = yaml.safe_load(data_file)

    for yaml_config in yaml_configs:
        stage_type = yaml_config.pop(CONFIG_STAGE_KEY)
        config = yaml_config

        try:
            stage = STAGE_TYPES[stage_type]  # stage_type comes from the yaml config
            logger.info(f"Loading {stage_type}...")
            stages.append(stage.from_config(**config))
        except TypeError as e:
            logger.error(f"Error encountered when loading {stage_type}: arguments '{config}' does not match "
                         f"required arguments for {stage_type}")
            raise e
        except FileNotFoundError as e:
            logger.error(f"Error encountered when loading {stage_type}: File in arguments for {stage_type} "
                         f"not found: '{config}'")
            raise e
        except Exception as e:
            logger.error(f"{type(e).__name__} encountered when loading {stage_type}: arguments '{config}'.")
            raise e

    return stages
