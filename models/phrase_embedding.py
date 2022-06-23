import torch
import torch.nn as nn
import os
import numpy as np
import pickle
import sys
import json

from config.config import CONF

sys.path.append(os.path.join(os.getcwd()))


class PhraseEmbeddingSent(torch.nn.Module):
    def __init__(self, phrase_embed_dim, use_bidir=False, emb_size=300):
        super(PhraseEmbeddingSent, self).__init__()

        self.emb_size = emb_size
        self.phrase_embed_dim = phrase_embed_dim
        self.use_bidir = use_bidir

        self.sg_anno = json.load(open(os.path.join(CONF.PATH.LANGUAGE, 'sg_anno.json'), 'r'))
        self.phrase_id = json.load(open(os.path.join(CONF.PATH.LANGUAGE, 'phrase_id.json'), 'r'))
        self.phrase_list = json.load(open(os.path.join(CONF.PATH.LANGUAGE, 'phrase_all.json'), 'r'))

        self.scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, 'ScanRefer_filtered_train.json'), 'r'))
        self.scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, 'ScanRefer_filtered_val.json'), 'r'))

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=phrase_embed_dim,
            batch_first=True,
            bidirectional=self.use_bidir
        )

    def forward(self, data_dict):
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)

        batch_phrase_embed = []
        batch_rel_phrase_embed = []
        batch_relation_conn = []
        batch_target_id = []

        glove = data_dict["glove"]

        object_id = data_dict["object_id"].cpu()
        batch_size = object_id.shape[0]
        sent_id = data_dict["sent_id"].cpu()
        index = data_dict["scan_idx"].cpu()
        phase = data_dict["phase"]

        for i in range(batch_size):
            phrase_embed, rel_phrase_embed, relation_conn, target_id = self.phrase_embed(
                index[i], object_id[i], sent_id[i], self.emb_size, phase, glove)

            batch_phrase_embed.append(phrase_embed)
            batch_rel_phrase_embed.append(rel_phrase_embed)
            batch_relation_conn.append(relation_conn)
            batch_target_id.append(target_id)

        return batch_phrase_embed, batch_rel_phrase_embed, batch_relation_conn, batch_target_id

    def phrase_embed(self, index, object_id, sent_id, emb_size, phase, glove):
        if phase == 'train':
            dataset = self.scanrefer_train
            scene_id = dataset[index]["scene_id"]
            sentence = dataset[index]["description"]
            object_name = dataset[index]["object_name"]
        else:
            dataset = self.scanrefer_val
            scene_id = dataset[index]["scene_id"]
            sentence = dataset[index]["description"]
            object_name = dataset[index]["object_name"]

        sent_sg = self.sg_anno[scene_id][str(object_id.numpy().tolist())][str(sent_id.numpy().tolist())]
        phrase_list = self.phrase_list[scene_id][str(object_id.numpy().tolist())][str(sent_id.numpy().tolist())]["phrase"]
        token_len = self.phrase_list[scene_id][str(object_id.numpy().tolist())][str(sent_id.numpy().tolist())]["token_len"]

        target = []
        for id in sent_sg:
            target.append(id[0])
        target_id = max(target, key=target.count)
        if object_name in sentence and object_name not in phrase_list[target_id]:
            count = 0
            tmp = []
            for i, w in enumerate(phrase_list):
                if object_name in w:
                    count = count+1
                    tmp.append(i)
            if count > 0:
                target_id = tmp[0]

        sen_phrase_embeds = []
        phrase_len = []
        phrase_split = []
        for p in phrase_list:
            m = str(p).split(' ')
            phrase_len.append(len(m))
            phrase_split.append(m)
            phrase_embeds_list = []
            for n in m:
                if n in glove:
                    phrase_embeds_list.append(torch.tensor(glove[n]))
                else:
                    phrase_embeds_list.append(torch.tensor(glove["unk"]))
            phrase_embeds = torch.zeros(len(m), 300)
            for i in range(len(m)):
                phrase_embeds[i, :] = phrase_embeds_list[i]
            _, phrase_embeds = self.gru(phrase_embeds.unsqueeze(0).cuda())
            phrase_embeds = phrase_embeds.permute(1, 0, 2).contiguous().flatten(start_dim=1)
            sen_phrase_embeds.append(phrase_embeds)
        phrase_embed = torch.cat(tuple(sen_phrase_embeds), 0)  # num_phrase*256

        relation_conn = []
        input_rel_phr_idx = []
        rel_lengths = []
        for rel_id, rel in enumerate(sent_sg):
            sbj_id, obj_id, rel_phrase = rel
            relation_conn.append([sbj_id, obj_id, rel_id])
            uni_rel_phr_idx = torch.zeros(token_len[0] + 5, emb_size)
            tokenized_phr_rel = rel_phrase.lower().split(' ')
            tokenized_phr_rel = phrase_split[sbj_id] + tokenized_phr_rel + phrase_split[obj_id]  # sbj + rel + obj
            rel_phr_len = len(tokenized_phr_rel)
            rel_lengths.append(rel_phr_len)
            for index, w in enumerate(tokenized_phr_rel):
                if w in glove:
                    uni_rel_phr_idx[index:index+1] = torch.tensor(glove[w])
                else:
                    uni_rel_phr_idx[index:index+1] = torch.tensor(glove["unk"])
            input_rel_phr_idx.append(uni_rel_phr_idx)

        if len(relation_conn) > 0:
            input_rel_phr_idx = torch.stack(input_rel_phr_idx).cuda()
            _, rel_phrase_embeds = self.gru(input_rel_phr_idx)
            rel_phrase_embed = rel_phrase_embeds.permute(1, 0, 2).contiguous().flatten(start_dim=1)
        else:
            rel_phrase_embed = None

        return phrase_embed, rel_phrase_embed, relation_conn, target_id