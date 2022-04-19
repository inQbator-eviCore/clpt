"""Abstract base class for NLP pipeline stages."""

import logging
from func_timeout import func_timeout, FunctionTimedOut
from abc import ABC, abstractmethod
from src.clao.clao import TextCLAO

logger = logging.getLogger(__name__)


class PipelineStage(ABC):

    @abstractmethod
    def __init__(self, timeout_seconds=1, **kwargs):
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, **kwargs):
        """Method invoked by pipeline_stage_creator"""
        return cls(**kwargs)

    @abstractmethod
    def process(self, clao_info: TextCLAO) -> None:
        """
        Apply this stage to the given CLAO info, updating the object as necessary
        Args:
            clao_info: The CLAO info to process
        Returns: None
        """
        pass

    def process_with_fallback(self, clao_info: TextCLAO) -> None:
        """
        Apply this stage's self.process() to the given CLAO info, falling back to self.fallback() if self.process()
        times out or returns another error
        Args:
            clao_info: The CLAO info to process
        Returns:
            None

        """
        class_name = '.'.join([self.__class__.__module__, self.__class__.__name__])
        try:
            func_timeout(self.timeout(), self.process, args=(clao_info,))
            return
        except FunctionTimedOut:
            logger.warning(f"CLAO {clao_info.name} timed out in main process for {class_name}, "
                           f"going to fallback.")
        except Exception as e:
            logger.warning(f"Failed to process {clao_info.name} in main process for {class_name}, "
                           f"going to fallback. {type(e).__name__} message: '{e}'")

        self.fallback(clao_info)

    def fallback(self, clao_info: TextCLAO) -> None:
        """
        If self.process() times out or returns another error, run this method instead
        Args:
            clao_info: The CLAO info to process

        Returns: None
        """
        return

    def timeout(self) -> int:
        """
        Returns: Time in seconds to wait for process to finish before falling back to self.fallback()
        """
        return self.timeout_seconds

    def __eq__(self, other: 'PipelineStage'):
        return isinstance(other, self.__class__)
