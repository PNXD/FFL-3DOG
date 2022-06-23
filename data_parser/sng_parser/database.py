# https://github.com/vacancy/SceneGraphParser

import os.path as osp
import spacy
import torch
from config.config import CONF

_caches = dict()


def load_list(filename):
    if filename not in _caches:
        out = set()
        for x in open(osp.join(osp.dirname(__file__), '_data', filename)):
            x = x.strip()
            if len(x) > 0:
                out.add(x)
        _caches[filename] = out
    return _caches[filename]


def load_own(filename):
    if filename not in _caches:
        out = set()
        for x in open(filename):
            x = x.strip()
            if len(x) > 0:
                out.add(x)
        _caches[filename] = out
    return _caches[filename]


def is_phrasal_verb(verb):
    return verb in load_list('phrasal-verbs.txt')


def is_phrasal_prep(prep):
    return prep in load_list('phrasal-preps.txt')


def is_scene_noun(noun):
    head = noun.split(' ')[-1]
    s = load_list('scene-nouns.txt') 
    return noun in s or head in s


def is_dir(word):
    return word in load_own('dir.txt')


def is_space(word):
    return word in load_own('spatial_relation.txt')


def relation(word):
    for i, j in enumerate(load_own('spa_complete.txt')):
        m = j.split(' ')
        if word in m:
            return j
    return word


def is_noun(chunks, i):
    for j, c in enumerate(chunks):
        if c.start <= i <= (c.end-1):
            return True


def locate_obj(chunks, i):
    m = list()
    for j, c in enumerate(chunks):
        if i < c.start:
            m.append(j)
    if len(m) == 0:
        return m
    if chunks[m[0]].root.lemma_ == '-PRON-':
        m.pop(0)
    return m


def replace(sent, chunks):
    if str(' bin ') in sent or str(' bin.') in sent or str(' bin,') in sent:
        nlp = spacy.load("en_core_web_lg")
    else:
        nlp = spacy.load("en_core_web_md")
    doc = nlp(sent)
    doc1 = []
    for token in doc:
        doc1.append(token)
    for i, entity in enumerate(chunks):
        if entity.root.lemma_ == '-PRON-'and i > 0:
            doc1[entity.root.i] = chunks[0]
        elif entity.root.lemma_ == '-PRON-'and i == 0:
            doc1[entity.root.i] = chunks[i+1]
    sent = " ".join('%s' % id for id in doc1)
    return sent


def delete_rela(list):
    if len(list) > 1:
        for index, i in enumerate(list):
            for index1, j in enumerate(list[(index+1):]):
                if i == j:
                    list[index + index1 + 1] = 0
        list1 = []
        for index, j in enumerate(list):
            if j == 0:
                list1.append(index)
        list1.reverse()
        for i in list1:
            list.pop(i)
    if len(list) > 1:
        for index, i in enumerate(list):
            for index1, j in enumerate(list[(index+1):]):
                if i['relation'] == j['relation'] and i['object'] == j['object']:
                    list[index] = 0
        list1 = []
        for index, j in enumerate(list):
            if j == 0:
                list1.append(index)
        list1.reverse()
        for i in list1:
            list.pop(i)

    for index, i in enumerate(list):
        if i['subject'] == i['object']:
            list[index] = 0
    list1 = []
    for index, j in enumerate(list):
        if j == 0:
            list1.append(index)
    list1.reverse()
    for i in list1:
        list.pop(i)
    return list


def is_adj(word):
    return word in load_own('un-adj.txt')


def sub(chunks, sent):
    s = [0]
    nlp = spacy.load("en_core_web_md")
    doc = nlp(sent)
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass'):
            for e, entity in enumerate(chunks):
                if entity.start <= token.i < entity.end:
                    s = []
                    s.append(e)
    return s


def correct_sbj(list, chunks, sent):
    s = sub(chunks, sent)
    for rela in list:
        if (rela['subject']+1) > len(chunks):
            rela['subject'] = s[0]
    return list


def unite_sbj(chunks, list, sent):
    s = sub(chunks, sent)
    for rela in list:
        if str(chunks[rela['subject']]) == str(chunks[s[0]]):
            rela['subject'] = s[0]
    return list
