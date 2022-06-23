#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : spacy_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/21/2018
#
# This file is part of SceneGraphParser.
# Distributed under terms of the MIT license.
# https://github.com/vacancy/SceneGraphParser
import spacy
from .. import database
from ..parser import Parser
from .backend import ParserBackend
import en_core_web_md

__all__ = ['SpacyParser']


@Parser.register_backend
class SpacyParser(ParserBackend):
    """
    Scene graph parser based on spaCy.
    """

    __identifier__ = 'spacy'

    def __init__(self, model="en_core_web_md"):
        """
        Args:
            model (str): a spec for the spaCy model. (default: en). Please refer to the
            official website of spaCy for a complete list of the available models.
            This option is useful if you are dealing with languages other than English.
        """

        self.model = model

        try:
            import spacy
        except ImportError as e:
            raise ImportError('Spacy backend requires the spaCy library. Install spaCy via pip first.') from e

        try:
            self.nlp = spacy.load(model)
        except OSError as e:
            raise ImportError('Unable to load the English model. Run `python -m spacy download en` first.') from e

    def parse(self, sentence, return_doc=False):
        """
        The spaCy-based parser parse the sentence into scene graphs based on the dependency parsing
        of the sentence by spaCy.

        All entities (nodes) of the graph come from the noun chunks in the sentence. And the dependencies
        between noun chunks are used for determining the relations among these entities.

        The parsing is performed in three steps:

            1. find all the noun chunks as the entities, and resolve the modifiers on them.
            2. determine the subject of verbs (including nsubj, acl and pobjpass). Please refer to the comments
            in the code for better explanation.
            3. determine all the relations among entities.
        """
        if str(' bin ') in sentence or str(' bin.') in sentence or str(' bin,') in sentence:
            nlp = spacy.load("en_core_web_lg")
            doc = nlp(sentence)
        else:
            doc = self.nlp(sentence)

        # Step 1: determine the entities.
        Doc = []
        for token in doc:
            Doc.append(token)
        entities = list()
        entity_chunks = list()
        for entity in doc.noun_chunks:
            # Ignore pronouns such as "it".
            if database.is_dir(entity.root.lemma_) == True and Doc[entity.end].dep_ != 'punct':
                continue
            if database.is_adj(entity.root.lemma_) == True and str(Doc[entity.root.i + 1]) == 'of':
                continue

            ent = dict(
                span=entity.text,
                lemma_span=entity.lemma_,
                head=entity.root.text,
                lemma_head=entity.root.lemma_,
                span_bounds=(entity.start, entity.end),
                modifiers=[]
            )

            for x in entity.root.children:
                # TODO(Jiayuan Mao @ 08/21): try to determine the number.
                if x.dep_ == 'det':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'lemma_span': x.lemma_})
                elif x.dep_ == 'nummod':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'lemma_span': x.lemma_})
                elif x.dep_ == 'amod':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'lemma_span': x.lemma_})
                elif x.dep_ == 'compound':
                    ent['head'] = x.text + ' ' + ent['head']
                    ent['lemma_head'] = x.lemma_ + ' ' + ent['lemma_head']

            if database.is_scene_noun(ent['lemma_head']):
                ent['type'] = 'scene'
            else:
                ent['type'] = 'unknown'

            entities.append(ent)
            entity_chunks.append(entity)

        # Step 2: determine the subject of the verbs.
        # To handle the situation where multiple nouns may be the same word,
        # the tokens are represented by their position in the sentence instead of their text.
        relation_subj = dict()
        for token in doc:
            # E.g., A [woman] is [playing] the piano.
            if token.dep_ == 'nsubj':
                relation_subj[token.head.i] = token.i
            # E.g., A [woman] [playing] the piano...
            elif token.dep_ == 'acl':
                relation_subj[token.i] = token.head.i
            # E.g., The piano is [played] by a [woman].
            elif token.dep_ == 'pobj' and token.head.dep_ == 'agent' and token.head.head.pos_ == 'VERB':
                relation_subj[token.head.head.i] = token.i

        # Step 3: determine the relations.
        relations = list()
        fake_noun_marks = set()

        for entity in doc.noun_chunks:
            # Again, the subjects and the objects are represented by their position.
            relation = None

            # E.g., A woman is [playing] the [piano].
            # E.g., The woman [is] a [pianist].
            if entity.root.dep_ in ('dobj', 'attr') and entity.root.head.i in relation_subj:
                relation = {
                    'subject': relation_subj[entity.root.head.i],
                    'object': entity.root.i,
                    'relation': entity.root.head.text,
                    'lemma_relation': entity.root.head.lemma_
                }
            elif entity.root.dep_ == 'pobj':
                # E.g., The piano is played [by] a [woman].
                if entity.root.head.dep_ == 'agent':
                    pass
                # E.g., A [woman] is playing with the piano in the [room].
                elif (
                        entity.root.head.head.pos_ == 'VERB' and
                        entity.root.head.head.i + 1 == entity.root.head.i and
                        database.is_phrasal_verb(entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_)
                ) and entity.root.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.head.text + ' ' + entity.root.head.text,
                        'lemma_relation': entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_
                    }
                # E.g., A [woman] is playing the piano in the [room]. Note that room.head.head == playing.
                # E.g., A [woman] playing the piano in the [room].
                elif (
                        entity.root.head.head.pos_ == 'VERB' or
                        entity.root.head.head.dep_ == 'acl'
                ) and entity.root.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.text,
                        'lemma_relation': entity.root.head.lemma_
                    }
                # E.g., A [woman] in front of a [piano].
                elif (
                        entity.root.head.head.dep_ == 'pobj' and
                        database.is_phrasal_prep(doc[entity.root.head.head.head.i:entity.root.head.i + 1].text.lower())
                ):
                    fake_noun_marks.add(entity.root.head.head.i)
                    relation = {
                        'subject': entity.root.head.head.head.head.i,
                        'object': entity.root.i,
                        'relation': doc[entity.root.head.head.head.i:entity.root.head.i + 1].text,
                        'lemma_relation': doc[entity.root.head.head.head.i:entity.root.head.i].lemma_
                    }
                # E.g., A [piano] in the [room].
                elif entity.root.head.head.pos_ == 'NOUN':
                    relation = {
                        'subject': entity.root.head.head.i,
                        'object': entity.root.i,
                        'relation': entity.root.head.text,
                        'lemma_relation': entity.root.head.lemma_
                    }
                # E.g., A [piano] next to a [woman].
                elif entity.root.head.head.dep_ in ('amod', 'advmod') and entity.root.head.head.head.pos_ == 'NOUN':
                    relation = {
                        'subject': entity.root.head.head.head.i,
                        'object': entity.root.i,
                        'relation': entity.root.head.head.text + ' ' + entity.root.head.text,
                        'lemma_relation': entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_
                    }
                # E.g., A [woman] standing next to a [piano].
                elif entity.root.head.head.dep_ in ('amod', 'advmod') and entity.root.head.head.head.pos_ == 'VERB' and entity.root.head.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.head.text + ' ' + entity.root.head.text,
                        'lemma_relation': entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_
                    }
                # E.g., A [woman] is playing the [piano] in the room
                elif entity.root.head.head.dep_== 'VERB' and entity.root.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.text,
                        'lemma_relation': entity.root.head.lemma_
                    }

            # E.g., The [piano] is played by a [woman].
            elif entity.root.dep_ == 'nsubjpass' and entity.root.head.i in relation_subj:
                # Here, we reverse the passive phrase. I.e., subjpass -> obj and objpass -> subj.
                relation = {
                    'subject': relation_subj[entity.root.head.i],
                    'object': entity.root.i,
                    'relation': entity.root.head.text,
                    'lemma_relation': entity.root.head.lemma_
                }

            if relation is not None:
                relations.append(relation)

        # Apply the `fake_noun_marks`.
        entities = [e for e, ec in zip(entities, entity_chunks) if ec.root.i not in fake_noun_marks]
        entity_chunk = [ec for ec in entity_chunks if ec.root.i not in fake_noun_marks]

        filtered_relations = list()
        for relation in relations:
            # Use a helper function to map the subj/obj represented by the position
            # back to one of the entity nodes.
            relation['subject'] = self.__locate_noun(entity_chunk, relation['subject'])
            relation['object'] = self.__locate_noun(entity_chunk, relation['object'])
            if relation['subject'] != None and relation['object'] != None:
                filtered_relations.append(relation)

        for token in doc:
            if database.is_space(str(token)) == True and Doc[token.i + 1].dep_ != 'punct':
                if database.is_noun(entity_chunks, token.i) == True:
                    continue
                if token.i < entity_chunks[0].start:
                    m = database.locate_obj(entity_chunks, token.i)
                    n = database.relation(str(token))
                    if len(m) != 0:
                        relation = {
                            'subject': m[0] + 1,
                            'object': m[0],
                            'relation': n,
                            'lemma_relation': str(token)
                        }
                        if relation is not None:
                            filtered_relations.append(relation)
                elif token.i >= entity_chunks[0].end:
                    m = database.locate_obj(entity_chunks, token.i)
                    n = database.relation(str(token))
                    s = database.sub(entity_chunks, sentence)
                    if len(m) != 0 and len(s) != 0:
                        if str(entity_chunks[s[0]]) != str(entity_chunks[m[0]]):
                            relation = {
                                'subject': s[0],
                                'object': m[0],
                                'relation': n,
                                'lemma_relation': str(token)
                            }
                            if relation is not None:
                                filtered_relations.append(relation)
                        if str(entity_chunks[s[0]]) == str(entity_chunks[m[0]]):
                            relation = {
                                'subject': m[0]-1,
                                'object': m[0],
                                'relation': n,
                                'lemma_relation': str(token)
                            }
                            if relation is not None:
                                filtered_relations.append(relation)

        for token in doc:
            if token.dep_ in ('ROOT', 'conj') and len(Doc[token.i: -1]) >= 3 and Doc[token.i + 1].dep_ == 'prep' and \
                    database.is_noun(entity_chunks, token.i + 2) == True and str(Doc[token.i + 1]) not in ('between',
                                                                                                           'beside'):
                m = database.locate_obj(entity_chunks, token.i)
                s = database.sub(entity_chunks, sentence)
                if len(m) != 0 and len(s) != 0:
                    relation = {
                        'subject': s[0],
                        'object': m[0],
                        'relation': str(Doc[token.i+1]),
                        'lemma_relation': str(Doc[token.i+1])
                    }
                    if relation is not None:
                        filtered_relations.append(relation)
            elif token.dep_ == 'prep' and entity_chunks[0].start == 1:
                if token.i == 0 or Doc[token.i - 1].dep_ == 'punct':
                    m = database.locate_obj(entity_chunks, token.i)
                    if len(m) != 0:
                        relation = {
                            'subject': m[0] + 1,
                            'object': m[0],
                            'relation': str(token),
                            'lemma_relation': str(token.lemma_)
                        }
                        if relation is not None:
                            filtered_relations.append(relation)
            elif token.dep_ == 'prep' and database.is_noun(entity_chunks, token.i+1) == True and database.is_noun(entity_chunks, token.i-1) == True:
                m = database.locate_obj(entity_chunks, token.i)
                if len(m) != 0:
                    relation = {
                        'subject': m[0]-1,
                        'object': m[0],
                        'relation': str(Doc[token.i]),
                        'lemma_relation': str(Doc[token.i])
                    }
                    if relation is not None:
                        filtered_relations.append(relation)

        filtered_relations = database.delete_rela(filtered_relations)
        filtered_relations = database.correct_sbj(filtered_relations, entity_chunks, sentence)
        filtered_relations = database.unite_sbj(entity_chunks, filtered_relations, sentence)

        if return_doc:
            return {'entities': entities, 'relations': filtered_relations}, doc
        return {'entities': entities, 'relations': filtered_relations}

    @staticmethod
    def __locate_noun(chunks, i):
        for j, c in enumerate(chunks):
            if c.start <= i < c.end:
                return j
        return None

    def __locate_obj(chunks, i):
        m = list()
        for j, c in enumerate(chunks):
            if i < c.start:
                m.append(j)
                return m
        return None

    def __locate_rela(chunks, word):
        for j, c in enumerate(chunks):
            return c.start <= word.i <= c.end
