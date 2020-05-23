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
import torch.nn.functional as F
from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
from knockoff.victim.blackbox import Blackbox
import knockoff.config as cfg
from knockoff.adversary import jacobian

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class JacobianAdversary(object):
    def __init__(self, queryset, blackbox, model, algo, eps, device, reward = 'all'):
        self.blackbox = blackbox
        self.queryset = queryset
        self.device = device
        # self.n_queryset = len(self.queryset)
        # self.batch_size = batch_size
        # self.idx_set = set()
        self.eps = eps
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
        self.algo = algo
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

    def get_transferset_adaptive(self, budget):

        # Implement the bandit gradients algorithm
        h_func = np.zeros(self.num_actions)
        learning_rate = np.zeros(self.num_actions)
        probs = np.ones(self.num_actions) / self.num_actions
        selected_x = []
        queried_labels = []

        avg_reward = 0
        actionListSelected = []
        pathCollection = []

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5, weight_decay=5e-4)
        criterion = model_utils.soft_cross_entropy

        originalVar = 0
        jacobianVar = 0
        adversarialVar = 0

        with tqdm(total=budget) as pbar:
            for iterate in range(1, budget+1):
                # Sample an action
                action = np.random.choice(np.arange(0, self.num_actions), p=probs)
                actionListSelected.append(action)
                # Sample data to attack
                sampled_x, path = self._sample_data(self.queryset, action)

                # """prepare inputs to MxNx3"""
                # sampled_x = np.rollaxis(sampled_x.cpu().numpy()[0], 0, 3)
                sampled_x = torch.tensor(sampled_x).to(self.device)

                """prepare one hot encoding target tensor input"""
                target = np.zeros(256)
                target[action-1] = 1
                # make target batch size = 1 by encapsulate it with a list
                target = torch.tensor([target]).to(self.device)

                """Query the local adversarial model"""
                self.model.eval()
                y_output = self.model(sampled_x)
                originalVar += self.printStats(y_output,action)


                """generate adversarial sample"""
                if self.algo == 'jsma':
                    # modification needed: target class != action
                    jacobian_input = jacobian.jsma(self.model, sampled_x, action).to(self.device)
                elif self.algo == 'fgsm':
                    jacobian_input = jacobian.fgsm(sampled_x, action, self.model, criterion, self.eps).to(self.device)
                else:
                    raise NotImplementedError

                jacobian_output = self.blackbox(jacobian_input)

                jacobianVar += self.printStats(jacobian_output, action)



                """Train the thieved classifier"""
                """to cuda"""
                # model = self.model.to('cuda')
                self.model.train()
                self.train(self.model, optimizer, criterion, jacobian_input, jacobian_output)

                """training knockoff nets for sampled data"""
                # Test new labels
                y_hat = self.model(jacobian_input)
                adversarialVar += self.printStats(y_hat, action)

                """Compute rewards"""
                reward = self._reward(jacobian_output.detach(), y_hat, iterate)
                avg_reward = avg_reward + (1.0 / iterate) * (reward - avg_reward)

                """Update learning rate"""
                learning_rate[action] += 1

                """Update H function"""
                for a in range(self.num_actions):
                    if a != action:
                        h_func[a] = h_func[a] + 1.0 / learning_rate[action] * (reward - avg_reward) * probs[a]
                    else:
                        h_func[a] = h_func[a] + 1.0 / learning_rate[action] * (reward - avg_reward) * (1 - probs[a])

                """Update probs"""
                aux_exp = np.exp(h_func)
                probs = aux_exp / np.sum(aux_exp)
                pbar.update()
                if max(probs) > 0.9:
                    code.interact(local=dict(globals(), **locals()))
                print(probs[action])
                print(np.partition(probs, -10)[-10:])
                print(set(list(learning_rate)))

                generated_sample = jacobian_input.detach().cpu()[0]
                # generated_sample = np.rollaxis(generated_sample, 0, 3)

                """prepare transferset"""
                selected_x.append((generated_sample, jacobian_output.cpu().squeeze().detach()))
                # Train the thieved classifier the final time???
            # model_utils.train_model(transferset)

            #
            # return thieved_classifier
        # print(probs)

        # code.interact(local=dict(globals(), **locals()))

        print(f"Original variance (from f'): {originalVar/budget:.10f} ｜ Jacobian Variance (from f): {jacobianVar/budget:.10f} | Adversarial Variance (for training f'): {adversarialVar/budget:.10f}")

        print(learning_rate)
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
        prev_avg = self.y_avg
        self.y_avg = self.y_avg + (1.0 / n) * (target[0].numpy() - self.y_avg)

        # Then compute reward
        # or try target[k]
        reward = 0
        for k in range(self.num_classes):
            reward += np.maximum(0, self.y_avg[k] - prev_avg[k])

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
        aux_exp = np.exp(output[0].detach().numpy())
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
        print(reward)
        return np.sum(reward)

    def printStats(self, output, label):
        _, pred_class = torch.max(output, 1)
        output = output.cpu()

        if hasattr(output, "grad"):
            output = output[0].detach().numpy()
        elif not isinstance(output, np.ndarray):
            output = output[0].numpy()

        top5 = [output[idx] for idx in np.argsort(output)[-5:][::1]]

        # print()
        # print("max prob: " + str(max(output)))
        # print("ave prob: " + str(np.average(output)))
        # print("variance: " + str(np.var(top5)))
        # print("difference: " + str(max(output) - np.average(output)))
        # print("original class:" + str(label))
        # print("predicted class:" + str(pred_class))
        # print("=============================")

        return np.var(top5)

    # def get_transferset(self, budget):
    #
    #     # Implement the bandit gradients algorithm
    #     h_func = np.zeros(self.num_actions)
    #     learning_rate = np.zeros(self.num_actions)
    #     probs = np.ones(self.num_actions) / self.num_actions
    #     selected_x = []
    #     queried_labels = []
    #
    #     avg_reward = 0
    #     actionListSelected = []
    #     pathCollection = []
    #
    #     optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5, weight_decay=5e-4)
    #     criterion = model_utils.soft_cross_entropy
    #
    #     originalVar = 0
    #     jacobianVar = 0
    #     adversarialVar = 0
    #
    #     with tqdm(total=budget) as pbar:
    #         for iterate in range(1, budget + 1):
    #             # Sample an action
    #             probs = np.ones(self.num_actions) / self.num_actions
    #             action = np.random.choice(np.arange(0, self.num_actions), p=probs)
    #             actionListSelected.append(action)
    #             # Sample data to attack
    #             sampled_x, path = self._sample_data(self.queryset, action)
    #
    #             # """prepare inputs to MxNx3"""
    #             # sampled_x = np.rollaxis(sampled_x.cpu().numpy()[0], 0, 3)
    #             sampled_x = torch.tensor(sampled_x).to(self.device)
    #
    #             """prepare one hot encoding target tensor input"""
    #             target = np.zeros(256)
    #             target[action - 1] = 1
    #             # make target batch size = 1 by encapsulate it with a list
    #             target = torch.tensor([target]).to(self.device)
    #
    #             """Query the local adversarial model"""
    #             self.model.eval()
    #             y_output = self.model(sampled_x)
    #             originalVar += self.printStats(y_output, action)
    #
    #             """generate adversarial sample"""
    #             if self.algo == 'jsma':
    #                 # modification needed: target class != action
    #                 jacobian_input = jacobian.jsma(self.model, sampled_x, action).to(self.device)
    #             elif self.algo == 'fgsm':
    #                 jacobian_input = jacobian.fgsm(sampled_x, action, self.model, criterion, self.eps).to(self.device)
    #             else:
    #                 raise NotImplementedError
    #
    #             jacobian_output = self.blackbox(jacobian_input)
    #
    #             jacobianVar += self.printStats(jacobian_output, action)
    #
    #             """Train the thieved classifier"""
    #             """to cuda"""
    #             # model = self.model.to('cuda')
    #             self.model.train()
    #             self.train(self.model, optimizer, criterion, jacobian_input, jacobian_output)
    #             #
    #             # """training knockoff nets for sampled data"""
    #             # # Test new labels
    #             # y_hat = self.model(jacobian_input)
    #             # adversarialVar += self.printStats(y_hat, action)
    #             #
    #             # """Compute rewards"""
    #             # reward = self._reward(jacobian_output.detach(), y_hat, iterate)
    #             # avg_reward = avg_reward + (1.0 / iterate) * (reward - avg_reward)
    #             #
    #             # """Update learning rate"""
    #             learning_rate[action] += 1
    #             #
    #             # """Update H function"""
    #             # for a in range(self.num_actions):
    #             #     if a != action:
    #             #         h_func[a] = h_func[a] + 1.0 / learning_rate[action] * (reward - avg_reward) * probs[a]
    #             #     else:
    #             #         h_func[a] = h_func[a] + 1.0 / learning_rate[action] * (reward - avg_reward) * (1 - probs[a])
    #             #
    #             # """Update probs"""
    #             # aux_exp = np.exp(h_func)
    #             # probs = aux_exp / np.sum(aux_exp)
    #
    #             if max(probs) > 0.9:
    #                 code.interact(local=dict(globals(), **locals()))
    #             print(np.partition(probs, -3)[-3:])
    #             print(set(list(learning_rate)))
    #
    #             generated_sample = jacobian_input.detach().cpu()[0]
    #             # generated_sample = np.rollaxis(generated_sample, 0, 3)
    #
    #             """prepare transferset"""
    #             selected_x.append((generated_sample, jacobian_output.cpu().squeeze().detach()))
    #             pbar.update()
    #             # Train the thieved classifier the final time???
    #         # model_utils.train_model(transferset)
    #
    #         #
    #         # return thieved_classifier
    #     # print(probs)
    #
    #     # code.interact(local=dict(globals(), **locals()))
    #
    #     print(
    #         f"Original variance (from f'): {originalVar / budget:.10f} ｜ Jacobian Variance (from f): {jacobianVar / budget:.10f} | Adversarial Variance (for training f'): {adversarialVar / budget:.10f}")
    #
    #     print(learning_rate)
    #     return selected_x

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

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5, weight_decay=5e-4)
        criterion = model_utils.soft_cross_entropy

        originalVar = 0
        jacobianVar = 0
        adversarialVar = 0

        with tqdm(total=budget) as pbar:
            for iterate in range(1, budget + 1):
                # Sample an action
                probs = np.ones(self.num_actions) / self.num_actions
                action = np.random.choice(np.arange(0, self.num_actions), p=probs)
                actionListSelected.append(action)
                # Sample data to attack
                sampled_x, path = self._sample_data(self.queryset, action)

                sampled_x = torch.tensor(sampled_x).to(self.device)

                """prepare one hot encoding target tensor input"""

                jacobian_output = self.blackbox(sampled_x)

                learning_rate[action] += 1

                print(set(list(learning_rate)))

                generated_sample = sampled_x.detach().cpu()[0]

                """prepare transferset"""
                selected_x.append((generated_sample, jacobian_output.cpu().squeeze().detach()))
                pbar.update()

        print(learning_rate)
        return selected_x


def main():
    parser = argparse.ArgumentParser(description='Construct apaptive transfer set')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct',
                        required=True)
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('--algo', metavar='ALGO', type=str, help='adversarial algorithm used to alter inputs' )
    parser.add_argument('--eps', metavar='e', type=float, help="epsilon for adversarial sample crafting", default = 0.5)

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
    parser.add_argument('--topk', type=int, help='The top k number of probabilities to retain', default=None)
    parser.add_argument('--adaptive', help='Whether to sample samples adaptively', choices=['True','False'], default=False,type=bool)

    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']+"-jacobian"
    knockoff_utils.create_dir(out_path)

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

    algo = params['algo']
    eps = params['eps']
    adaptive = params["adaptive"]
    adversary = JacobianAdversary(queryset, blackbox, model, algo, eps, device,reward = 'all')

    print('=> constructing transfer set...')
    if adaptive:
        transferset = adversary.get_transferset_adaptive(params['budget'])
    else:
        transferset = adversary.get_transferset(params['budget'])
    if defense:
        transfer_out_path = osp.join(out_path, algo+'-'+defense+'-transferset.pickle')
    else:
        transfer_out_path = osp.join(out_path, algo + '-transferset.pickle')
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