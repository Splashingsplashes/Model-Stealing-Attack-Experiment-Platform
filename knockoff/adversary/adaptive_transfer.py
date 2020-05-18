#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import code
from torch.autograd import Variable
import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime
import knockoff.models.zoo as zoo
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
from knockoff.victim.blackbox import Blackbox
import knockoff.config as cfg

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class AdaptiveAdversary(object):
    def __init__(self, queryset, blackbox, model, device, reward = 'all'):
        self.blackbox = blackbox
        self.queryset = queryset
        self.device = device
        # self.n_queryset = len(self.queryset)
        # self.batch_size = batch_size
        # self.idx_set = set()

        self.num_classes = len(set(self.queryset.targets))
        self.num_actions = len(set(self.queryset.targets))

        self.reward = reward
        self.transferset = []  # List of tuples [(img_path, output_probs)]
        self.model = model
        if self.reward == 'div' or self.reward == 'all':
            self.y_avg = np.zeros(self.num_classes)

        # We need to keep an average and variance version of rewards
        # if self.reward == 'all':
        #     self.reward_avg = np.zeros(3)
        #     self.reward_var = np.zeros(3)

        self.reward_avg = np.zeros(3)
        self.reward_var = np.zeros(3)

    #     self._restart()
    #
    # def _restart(self):
    #     np.random.seed(cfg.DEFAULT_SEED)
    #     torch.manual_seed(cfg.DEFAULT_SEED)
    #     torch.cuda.manual_seed(cfg.DEFAULT_SEED)
    #
    #     self.idx_set = set(range(len(self.queryset)))
    #     self.transferset = []

    def get_transferset(self, budget):

        # Implement the bandit gradients algorithm
        h_func = np.zeros(self.num_actions)
        learning_rate = np.zeros(self.num_actions)
        probs = np.ones(self.num_actions) / self.num_actions
        selected_x = []
        queried_labels = []

        avg_reward = 0
        actionListSelected = []
        pathCollection = []
        class_count = np.zeros(self.num_actions)
        with tqdm(total=budget) as pbar:
            for iterate in range(1, budget+1):
                # Sample an action
                action = np.random.choice(np.arange(0, self.num_actions), p=probs)

                class_count[action] += 1

                actionListSelected.append(action)
                # Sample data to attack
                sampled_x, path = self._sample_data(self.queryset, action)

                # Query the victim classifier
                """to cuda"""
                sampled_x = sampled_x.to(self.device)
                y_output = self.blackbox(sampled_x)
                # code.interact(local=dict(globals(), **locals()))

                # fake_label = np.argmax(y_output, axis=1)
                # fake_label = to_categorical(labels=fake_label, nb_classes=self.classifier.nb_classes())

                queried_labels.append(y_output.cpu())

                # Train the thieved classifier
                self.model.train()

                optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5, weight_decay=5e-4)
                criterion = model_utils.soft_cross_entropy

                y_output = y_output.to(self.device)
                self.train(self.model, optimizer, criterion, sampled_x, y_output)

                # Test new labels
                self.model.eval()
                y_hat = self.model(sampled_x)

                sampled_x = sampled_x.cpu().numpy()[0]
                # code.interact(local=dict(globals(), **locals()))
                # sampled_x = np.transpose(sampled_x)
                sampled_x = np.rollaxis(sampled_x, 0, 3)
                code.interact(local=dict(globals(), **locals()))
                selected_x.append((sampled_x, y_output.cpu().squeeze().detach()))

                pathCollection.append((path[0], y_output.detach().cpu().squeeze()))
                # Compute rewards
                reward = self._reward(y_output.detach(), y_hat.detach().cpu(), iterate)
                avg_reward = avg_reward + (1.0 / iterate) * (reward - avg_reward)

                # Update learning rate
                learning_rate[action] += 1

                # Update H function
                # for a in range(self.num_actions):
                #     if a != action:
                #         h_func[a] = h_func[a] - 1.0 / learning_rate[action] * (reward - avg_reward) * probs[a]
                #     else:
                #         h_func[a] = h_func[a] + 1.0 / learning_rate[action] * (reward - avg_reward) * (1 - probs[a])

                for a in range(self.num_actions):
                    if a != action:
                        h_func[a] = h_func[a] + (1.0 / learning_rate[action]) * (reward - avg_reward) * probs[a]
                    else:
                        h_func[a] = h_func[a] + (1.0 / learning_rate[action]) * (reward - avg_reward) * (1 - probs[a])

                # Update probs
                aux_exp = np.exp(h_func)
                probs = aux_exp / np.sum(aux_exp)
                # code.interact(local=dict(globals(), **locals()))
                pbar.update()
                print(class_count)
                # Train the thieved classifier the final time???
            # model_utils.train_model(transferset)

            #
            # return thieved_classifier
        # print(probs)
        print(class_count)
        return selected_x

    def train(self, model, optimizer, criterion, sampled_x, y_output):
        optimizer.zero_grad()
        outputs = model(sampled_x)
        # code.interact(local=dict(globals(), **locals()))
        loss = criterion(outputs, y_output)
        loss.backward()
        optimizer.step()

    def _sample_data(self, queryset, action):
        # x = [queryset[idx][0] for idx in range(len(queryset)) if queryset.targets[idx] == action]
        x=[]
        path=[]
        tensor = None

        try:
            for idx in range(len(queryset)):
                if self.queryset.targets[idx] == action:
                    tensor = queryset[idx][0]
                    tensor = tensor.unsqueeze(0)
                    x.append(tensor)
                    path.append(queryset.samples[idx])

            rnd_idx = np.random.choice(len(x))
        except ValueError:
            print('action = ' + str(action))
            code.interact(local=dict(globals(), **locals()))
        return x[rnd_idx], path[rnd_idx]

    def _reward(self, target, output, n):
        if self.reward == 'cert':
            return self._reward_cert(target)
        elif self.reward == 'div':
            return self._reward_div(target, n)
        elif self.reward == 'loss':
            return self._reward_loss(target, output)
        else:
            return self._reward_all(target, output, n)

    def _reward_cert(self, target):
        largests = np.partition(target.cpu().flatten(), -2)[-2:]
        reward = largests[1] - largests[0]

        return reward

    def _reward_div(self, target, n):
        # First update y_avg
        # or try target
        # code.interact(local=dict(globals(), **locals()))
        target = target.cpu()
        self.y_avg = self.y_avg + (1.0 / n) * (target[0].numpy() - self.y_avg)

        # Then compute reward
        # or try target[k]
        reward = 0
        for k in range(self.num_classes):
            reward += np.maximum(0, target[0][k] - self.y_avg[k])

        return reward

    def _reward_loss(self, target, output):
        # Compute victim probs
        # or try target
        target = target.cpu()
        output = output.cpu()
        aux_exp = np.exp(target[0])
        # code.interact(local=dict(globals(), **locals()))
        probs_output = aux_exp / np.sum(aux_exp.numpy())

        # Compute thieved probs
        output = output[0].detach().numpy()
        output = output.astype('float128')
        aux_exp = np.exp(output)
        probs_hat = aux_exp / np.sum(aux_exp)

        # Compute reward, cross entropy loss
        reward = 0
        for k in range(self.num_classes):
            reward += -probs_output[k] * np.log(probs_hat[k])

        return reward

    def _reward_all(self, y_output, y_hat, n):

        reward_cert = self._reward_cert(y_output)
        reward_div = self._reward_div(y_output, n)
        reward_loss = self._reward_loss(y_output, y_hat)
        reward = [reward_cert, reward_div, reward_loss]


        temp = np.subtract(reward, self.reward_avg)
        self.reward_avg = self.reward_avg + (1.0 / n) * temp

        temp = np.subtract(reward, self.reward_avg)
        self.reward_var = self.reward_var + (1.0 / n) * (temp ** 2 - self.reward_var)

        # Normalize rewards: z-score
        if n > 1:
            reward = temp / np.sqrt(self.reward_var)
        else:
            reward = [max(min(r, 1), 0) for r in reward]

        #IBM used np.mean()

        return np.sum(reward)



def main():
    parser = argparse.ArgumentParser(description='Construct apaptive transfer set')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct',
                        required=True)
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')

    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)
    # parser.add_argument('--topk', metavar='N', type=int, help='Use posteriors only from topk classes',
    #                     default=None)
    # parser.add_argument('--rounding', metavar='N', type=int, help='Round posteriors to these many decimals',
    #                     default=None)
    # parser.add_argument('--tau_data', metavar='N', type=float, help='Frac. of data to sample from Adv data',
    #                     default=1.0)
    # parser.add_argument('--tau_classes', metavar='N', type=float, help='Frac. of classes to sample from Adv data',
    #                     default=1.0)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--defense', type=str, help='Defense strategy used by victim side', default=None)

    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path+"-adaptive")
    transfer_out_path = osp.join(out_path+"-adaptive", 'transferset.pickle')

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up queryset
    queryset_name = params['queryset']
    valid_datasets = datasets.__dict__.keys()
    if queryset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[queryset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    queryset = datasets.__dict__[queryset_name](train=True, transform=transform)
    queryset.targets = [queryset.samples[idx][1] for idx in range(len(queryset))]
    # code.interact(local=dict(globals(), **locals()))
    num_classes = len(queryset.classes)
    # code.interact(local=dict(globals(), **locals()))
    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    defense = params['defense']
    blackbox = Blackbox.from_modeldir(blackbox_dir, defense, device)

    # ----------- Initialize Knockoff Nets
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    model = model.to(device)

    adversary = AdaptiveAdversary(queryset, blackbox, model, device, reward = 'all')

    print('=> constructing transfer set...')
    transferset = adversary.get_transferset(params['budget'])


    with open(transfer_out_path, 'wb') as wf:
        pickle.dump(transferset, wf)
    print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params_transfer.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()