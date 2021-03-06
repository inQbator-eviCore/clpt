"""Check spelling and correct any spelling error."""
from spellchecker import SpellChecker

from src.clao.text_clao import TextCLAO, Token
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import SPELL_CORRECTED_TOKEN


class SpellCorrectLevenshtein(PipelineStage):
    """Correct spelling using Levenshtein Distance."""

    def __init__(self, **kwargs):
        """Levenshtein Distance for spell correction."""
        super(SpellCorrectLevenshtein, self).__init__(**kwargs)
        # By default, distance=2 and this refers to edit_distance_2 method which computes all strings that are
        # two edits away from `word` using only the letters in the corpus
        self.spell = SpellChecker()

        # For longer words, it is highly recommended to use a distance of 1 and not the default 2.
        # edit_distance_1 method computes all strings that are one edit away from `word` using only the letters
        # in the corpus
        self.spell_long = SpellChecker(distance=1)

    def process(self, clao_info: TextCLAO):
        """Correct spelling using Levenshtein Distance.

        The default parameter in the SpellChecker() is distance=2 which refers to edit_distance_2 method that computes
        all strings that are two edits away from `word` using only the letters in the corpus. If the token length is
        greater than 14, we consider this to be a long word and then will use distance=1, as for longer words, it is
        highly recommended to use a distance of 1 and not the default 2.

        Args:
            clao_info (TextCLAO): the CLAO information to process
        """
        # based on the reference for word length frequency, length>=14 is set as the threshold for long words
        # https://math.wvu.edu/~hdiamond/Math222F17/Sigurd_et_al-2004-Studia_Linguistica.pdf

        # TODO: optimize the threshold for spell check if the token is long by modifying the spell.distance
        for token in clao_info.get_annotations(Token):
            if len(token.text) >= 14:
                token.map[SPELL_CORRECTED_TOKEN] = self.spell_long.correction(token.text)
            else:
                token.map[SPELL_CORRECTED_TOKEN] = self.spell.correction(token.text)
