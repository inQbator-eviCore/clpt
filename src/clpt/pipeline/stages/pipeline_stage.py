"""Abstract base class for NLP pipeline stages."""

import logging
from abc import ABC, abstractmethod
from typing import List
import nltk
from func_timeout import FunctionTimedOut, func_timeout
from src.clao.text_clao import TextCLAO

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Abstract base class for NLP pipeline stages.

    Properties:
        single_clao: True if this stage is meant to operate on one single CLAO at a time. False if it operates on all
        CLAOs. The default is True.
    """
    @abstractmethod
    def __init__(self, timeout_seconds=1, **kwargs):
        """Abstract base class for NLP pipeline stages.

        Args:
            timeout_seconds: seconds until stage is considered timed out. The default is 1
        """
        self.timeout_seconds = timeout_seconds
        self.single_clao = True  # TODO handle this much better

    @classmethod
    def from_config(cls, **kwargs):
        """Method invoked by pipeline_stage_creator"""
        return cls(**kwargs)

    @abstractmethod
    def process(self, clao_info: TextCLAO) -> None:
        """Apply this stage to the given CLAO info, updating the object as necessary.

        Args:
            clao_info: the CLAO info to process

        Returns:
            None
        """
        pass

    def process_with_fallback(self, clao_info: TextCLAO) -> None:
        """Apply this stage's self.process() to the given CLAO info, falling back to self.fallback() if self.process()
        times out or returns another error.

        Args:
            clao_info: the CLAO info to process

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
        """If self.process() times out or returns another error, run this method instead.

        Args:
            clao_info: the CLAO info to process

        Returns:
            None
        """
        return

    def timeout(self) -> int:
        """Return: Time in seconds to wait for process to finish before falling back to self.fallback()."""
        return self.timeout_seconds

    def __eq__(self, other: 'PipelineStage'):
        return isinstance(other, self.__class__)


class NltkStage(PipelineStage):
    def __init__(self, nltk_reqs: List, **kwargs):
        """Add to the pipeline class using the NLTK stages.

        Args:
            nltk_reqs (list): a list of files to be found in the NLTK Data Package
        """
        super(NltkStage, self).__init__(**kwargs)
        for req in nltk_reqs:
            try:
                nltk.data.find(req)
            except LookupError:
                nltk.download(req.split('/')[-1])

    @abstractmethod
    def process(self, clao_info: TextCLAO) -> None:
        """Process CLAO information."""
        pass
