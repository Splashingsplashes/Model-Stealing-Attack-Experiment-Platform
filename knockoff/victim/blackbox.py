#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import json
import code
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from knockoff.utils.type_checks import TypeCheck
import knockoff.utils.model as model_utils
import knockoff.models.zoo as zoo
from knockoff import datasets

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class Blackbox(object):
    def __init__(self, model, defense, device=None, output_type='probs', topk = None, rounding=None, ):
        self.device = torch.device('cuda') if device is None else device
        self.output_type = output_type

        self.defense = defense
        self.topk = topk
        self.rounding = rounding
        self.model = model.to(device)
        self.output_type = output_type
        self.model.eval()

        self.__call_count = 0

    @classmethod
    def from_modeldir(cls, model_dir, defense, device=None, output_type='probs'):
        device = torch.device('cuda') if device is None else device

        # What was the model architecture used by this model?
        params_path = osp.join(model_dir, 'params.json')
        with open(params_path) as jf:
            params = json.load(jf)
        model_arch = params['model_arch']
        num_classes = params['num_classes']
        victim_dataset = params.get('dataset', 'imagenet')
        modelfamily = datasets.dataset_to_modelfamily[victim_dataset]

        # Instantiate the model
        # model = model_utils.get_net(model_arch, n_output_classes=num_classes)
        model = zoo.get_net(model_arch, modelfamily, pretrained=None, num_classes=num_classes)
        model = model.to(device)

        # Load weights
        checkpoint_path = osp.join(model_dir, 'model_best.pth.tar')
        if not osp.exists(checkpoint_path):
            checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))

        blackbox = cls(model, defense, device, output_type)
        return blackbox

    def truncate_output(self, y_t_probs):
        if self.topk is not None:
            # Zero-out everything except the top-k predictions
            topk_vals, indices = torch.topk(y_t_probs, self.topk)
            newy = torch.zeros_like(y_t_probs)
            if self.rounding == 0:
                # argmax prediction
                newy = newy.scatter(1, indices, torch.ones_like(topk_vals))
            else:
                newy = newy.scatter(1, indices, topk_vals)
            y_t_probs = newy

        # Rounding of decimals
        if self.rounding is not None:
            y_t_probs = torch.Tensor(np.round(y_t_probs.numpy(), decimals=self.rounding))

        if self.defense == 'flatten':
            topk_vals, indices = torch.topk(y_t_probs, 5)
            code.interact(local=dict(globals(), **locals()))
            topk_vals = topk_vals.cpu().numpy()
            avg = sum(topk_vals)/5
            y_t_probs[indices[0]] = avg + 0.000001
            y_t_probs[indices[1]] = avg + 0.0000005
            y_t_probs[indices[2]] = avg
            y_t_probs[indices[3]] = avg - 0.0000005
            y_t_probs[indices[4]] = avg - 0.000001
            code.interact(local=dict(globals(), **locals()))
            # top5 = [y_t_probs[idx] for idx in np.argsort(y_t_probs)[-5:][::1]]




        return y_t_probs

    def __call__(self, query_input):
        TypeCheck.multiple_image_blackbox_input_tensor(query_input)

        # with torch.no_grad():
        #     query_input = query_input.to(self.device)
        #     query_output = self.model(query_input)
        #     self.__call_count += query_input.shape[0]
        #
        #     query_output_probs = F.softmax(query_output, dim=1).cpu()

        query_input = query_input.to(self.device)
        query_output = self.model(query_input)
        self.__call_count += query_input.shape[0]

        query_output_probs = F.softmax(query_output, dim=1).cpu()
        query_output_probs = self.truncate_output(query_output_probs)


        return query_output_probs
