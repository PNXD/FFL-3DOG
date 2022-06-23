import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.getcwd()))


class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False, hidden_size=128):
        super().__init__() 

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir

        lang_size = hidden_size * 2 if self.use_bidir else hidden_size

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes),
                nn.Dropout()
            )

    def forward(self, data_dict):
        """
        encode the input descriptions
        """
        # get the encoded language features
        lang_last = data_dict["batch_target_embed"]  # batch_size*hidden_size
        
        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(lang_last)  # B*num_text_classes

        return data_dict

