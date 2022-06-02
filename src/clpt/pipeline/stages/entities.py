"""NLP classification stage for extracting entities and performing Mention Detection task."""
import logging
from typing import List

import medspacy
import yaml
from blist import blist
from medspacy.target_matcher import TargetRule
from spacy.tokens import Doc

from src.clao.text_clao import Entity, EntityGroup, Sentence, TextCLAO
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.constants.annotation_constants import ENTITY_GROUP, EntityType

OOV = '<OOV>'

logger = logging.getLogger(__name__)


class MentionDetection(PipelineStage):
    """Detect mentions.

    Attributes:
        rules_file: a file that contains the rule set used for Mention Detection. In the default configs, this file is
        `src/resources/mention-detection-rules.yaml`.
    """
    def __init__(self, rules_file, **kwargs):
        """Load medspacy model."""
        super(MentionDetection, self).__init__(**kwargs)
        # Load medspacy model
        self.nlp = medspacy.load()
        with open(rules_file) as f:
            rules_dict = yaml.safe_load(f)
        self.custom_rules = rules_dict

    def process(self, clao_info: TextCLAO) -> None:
        """Extract mention based on the rules and add the extracted entities into CLAO(s) with the attributes of each of
        entities.

        Args:
            clao_info: the CLAO information to process
        """
        entity_type = EntityType.MENTION

        # Add rules for target concept extraction
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        target_rules = [TargetRule(**rule) for rule in self.custom_rules]
        target_matcher.add(target_rules)

        entity_index_offset = len(clao_info.get_annotations(Entity))
        sents: List[Sentence] = clao_info.get_annotations(Sentence)
        all_entities = blist()
        for sent in sents:
            token_id_offset = sent._token_id_range[0]
            doc = Doc(self.nlp.vocab, [token.text for token in sent.tokens])
            for pipe in self.nlp.pipeline:
                pipe[1](doc)
            if doc.ents:
                entities = blist()
                for i, ent in enumerate(doc.ents):
                    token_start = token_id_offset + ent.start
                    token_end = token_id_offset + ent.end
                    text = ent._.literal
                    entity = Entity(entity_index_offset + i, entity_type, 1, text, ent.label_, (token_start, token_end),
                                    clao_info)
                    entities.append(entity)
                entity_index_offset += len(entities)
                sent._entity_id_range = (entities[0].element_id, entities[-1].element_id + 1)
                all_entities.extend(entities)
        clao_info.insert_annotations(Entity, all_entities)


class GroupEntities(PipelineStage):
    """Group entities into EntityGroup.

    Attributes:
        entity_type (EntityType): the type of entity to group
    """
    def __init__(self, entity_type: str, **kwargs):
        self.entity_type = EntityType(entity_type)
        super(GroupEntities, self).__init__(**kwargs)

    def process(self, clao_info: TextCLAO) -> None:
        """Extract entities and group relevant entities into a entity group and add entity group to CLAO.

        Args:
            clao_info (TextCLAO): the CLAO information to process
        """
        entities = clao_info.get_annotations(Entity)
        entity_group_id_offset = len(clao_info.get_annotations(EntityGroup))
        entity_groups = {}
        for entity in entities:
            entity_tuples = (entity.literal, entity.label)
            if entity.entity_type == self.entity_type:
                if entity_tuples not in entity_groups:
                    entity_groups[entity_tuples] = EntityGroup(entity_group_id_offset + len(entity_groups),
                                                               self.entity_type, entity.literal, entity.label)

                entity.map[ENTITY_GROUP] = entity_groups[entity_tuples].element_id

        clao_info.insert_annotations(EntityGroup, list(entity_groups.values()))
