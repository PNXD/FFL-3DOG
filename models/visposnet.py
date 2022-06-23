# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Visual Positioning Network's framework
"""

import torch
import sys
import os
import numpy as np
from torch import nn
from models.lanref import lanref as GroundingModule
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.lang_module import LangModule
from utils.model_util_scannet import ScannetDatasetConfig

DC = ScannetDatasetConfig()

sys.path.append(os.path.join(os.getcwd()))


class VisPosNet(nn.Module):
    """
    Main class for Visual Positioning Network.
    It consists of five main parts:
    - backbone
    - voting
    - proposal
    - language guided grounding
    - language
    """

    def __init__(self, input_feature_dim, mean_size_arr, num_proposal=128, use_bidir=False, no_reference=False,
                 no_detection=False, use_lang_classifier=True, num_class=18, num_heading_bin=1, num_size_cluster=18,
                 sampling="vote_fps", vote_factor=1):
        super(VisPosNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.mean_size_arr = mean_size_arr
        self.num_size_cluster = num_size_cluster
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.num_proposal = num_proposal
        self.use_bidir = use_bidir
        self.no_reference = no_reference
        self.no_detection = no_detection
        self.use_lang_classifier = use_lang_classifier
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.sampling = sampling
        self.vote_factor = vote_factor

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal,
                                       sampling)

        # Language Guided Grounding
        if not no_reference:
            self.vg_head = GroundingModule(self.input_feature_dim, use_bidir, use_lang_classifier, num_proposal)

            # language classification
            if use_lang_classifier:
                self.lang = LangModule(num_class, use_lang_classifier, use_bidir)

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud must
                    be formated as (x, y, z, features...)
        Returns:
            data_dict: dict
        """

        # --------- BACKBONE MODULE ---------
        data_dict = self.backbone_net(data_dict)

        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features

        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)

        if not self.no_reference:
            # --------- GENERATE PRE_BOX ---------
            pred_center = data_dict["center"]  # B,num_proposal,3
            pred_heading_class = torch.argmax(data_dict["heading_scores"], -1)  # B,num_proposal
            pred_heading_residual = torch.gather(data_dict["heading_residuals"], 2,
                                                 pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
            pred_heading_residual = pred_heading_residual.squeeze(2)
            pred_size_class = torch.argmax(data_dict["size_scores"], -1)  # B,num_proposal
            pred_size_residual = torch.gather(data_dict["size_residuals"], 2, pred_size_class.unsqueeze(-1).
                                              unsqueeze(-1).repeat(1, 1, 1, 3))  # B,num_proposal,1,3
            pred_size_residual = pred_size_residual.squeeze(2)  # B,num_proposal,3
            pred_bbox = DC.param2bbox(pred_center[:, :, 0:3], pred_heading_class[:, :], pred_heading_residual[:, :],
                                       pred_size_class[:, :], pred_size_residual[:, :])

            data_dict["precomp_boxes"] = pred_bbox

            # --------- TARGET GROUNDING ---------
            data_dict = self.vg_head(data_dict)

            # --------- LANGUAGE ENCODING ---------
            if self.use_lang_classifier:
                data_dict = self.lang(data_dict)

        return data_dict
