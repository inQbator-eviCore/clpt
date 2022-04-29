import re
from typing import List

from omegaconf import DictConfig, OmegaConf, open_dict

from src.clao.pipeline_documents import Span


def add_new_key_to_cfg(cfg: DictConfig, value: str, *keys: str) -> None:
    """Add value to config section following key path, where key(s) do not already exist in config

    Args:
        cfg: config to add value to
        value: value to add to config
        *keys: config section names pointing to the desired key. Final item in list will be given value of value

    Returns: None

    """
    cfg_section = cfg
    for key in keys[:-1]:
        cfg_section = cfg_section[key]
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg_section[keys[-1]] = value


def match(regex: re.Pattern, string: str, offset: int, keep_between: bool) -> List[Span]:
    """
    Splits given input string into a List of Spans based on provided regex
    Args:
        regex: Compiled regular expression object
        string: The string to split
        offset: character index of the document that the input text starts at
        keep_between: true if text between matches should be included in list, else false

    Returns:
        A list of Spans split from the input text by the regex
    """
    span_array = []
    end_index = len(string)
    cursor = 0

    while cursor < end_index:
        for m in regex.finditer(string):
            start = m.start()
            if keep_between and start > cursor:
                span_array.append(Span(offset + cursor, offset + start))

            end = m.end()
            span_array.append(Span(offset + start, offset + end))
            cursor = end

        text = string[cursor:]
        if len(text.strip()) > 0:
            span_array.append(Span(offset + cursor, offset + end_index))

        cursor = end_index

    return span_array
