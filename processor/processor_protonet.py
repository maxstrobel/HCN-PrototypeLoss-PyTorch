import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from processor.processor import Processor
from processor.utils.utils import DictAction, euclidean_dist


class ProcessorProtoNet(Processor):
    """
    Processor for training with prototype loss; adapted:
        * data sampling
        * training / loss function
        * testing
        * added some parser options
    """
    def __init__(self, argv):
        super().__init__(argv)

    def load_data(self):
        """
        Load data with special sampler for training of prototype loss
        """
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        if 'debug' not in self.arg.test_feeder_args:
            self.arg.test_feeder_args['debug'] = self.arg.debug

        data_loader = dict()
        if self.arg.phase == 'train':
            feeder_train = self.fileio.load_feeder(self.arg.feeder, **self.arg.train_feeder_args)
            sampler_train = self.fileio.load_sampler(self.arg.sampler, labels=feeder_train.get_label(),
                                                     **self.arg.train_sampler_args)
            data_loader['train'] = DataLoader(
                dataset=feeder_train,
                batch_sampler=sampler_train,
                num_workers=self.arg.num_worker)
            self.logger.print_log(f'DataLoader: {len(data_loader["train"].dataset)} training samples loaded')
        if self.arg.test_feeder_args:
            feeder_test = self.fileio.load_feeder(self.arg.feeder, **self.arg.test_feeder_args)
            sampler_test = self.fileio.load_sampler(self.arg.sampler, labels=feeder_test.get_label(),
                                                    **self.arg.test_sampler_args)
            data_loader['test'] = DataLoader(
                dataset=feeder_test,
                batch_sampler=sampler_test,
                num_workers=self.arg.num_worker)
            self.logger.print_log(f'DataLoader: {len(data_loader["test"].dataset)} test samples loaded')

        return data_loader

    def train(self):
        """
        Train model an epoch using the Prototype loss procedure
        """
        n_class = self.arg.train_sampler_args['num_classes']
        n_support = self.arg.train_sampler_args['num_support']
        n_query = self.arg.train_sampler_args['num_query']

        self.model.train()
        loader = self.data_loader['train']
        loss_value = []

        with tqdm(total=len(loader)) as t:
            for data, label in loader:
                # get data
                data = data.float().to(self.dev)

                # forward
                z = self.model(data)
                z_support = z[:n_class * n_support]
                z_query = z[-n_class * n_query:]
                loss, distances = self.loss(z_query, z_support, n_class, n_support, n_query)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # statistics
                self.iter_info['loss'] = loss.data.item()
                self.iter_info['learning rate'] = self.optimizer.param_groups[0]["lr"]
                loss_value.append(self.iter_info['loss'])
                self.show_iter_info(t)
                self.meta_info['iter'] += 1

        self.epoch_info['mean_loss_train'] = np.mean(loss_value)
        self.show_epoch_info()

    def test(self):
        """
        Testing model using the Prototype loss procedure
        """
        n_class = self.arg.test_sampler_args['num_classes']
        n_support = self.arg.test_sampler_args['num_support']
        n_query = self.arg.test_sampler_args['num_query']

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        prediction_frag = []
        result_frag = []
        label_frag = []

        with tqdm(total=len(loader)) as t:
            for data, label in loader:
                # get data
                data = data.float().to(self.dev)
                label = label.float().to(self.dev)

                # inference
                with torch.no_grad():
                    z = self.model(data)
                    z_support = z[:n_class * n_support]
                    z_query = z[-n_class * n_query:]
                    loss, distances = self.loss(z_query, z_support, n_class, n_support, n_query)

                # statistics
                class_labels = label[:n_class * n_support:n_support]  # Determine labels of run
                _, distance_order = distances.sort()  # get indexes for downwardly sorted probabilities
                if isinstance(self.loss, KNNLoss):
                    distance_order = distance_order // n_support
                prediction = class_labels[distance_order]  # get corresponding class labels

                result_frag.append(z_query.data.cpu().numpy())
                loss_value.append(loss)
                label_frag.append(label[-n_class * n_query:].data.cpu().numpy().reshape(-1))
                prediction_frag.append(prediction.data.cpu().numpy().reshape(n_class * n_query, -1))

                self.iter_info['loss'] = loss.data.item()
                self.update_progress_bar(t)

        # evaluation
        self.result['output'] = np.concatenate(result_frag)
        self.result['prediction'] = np.concatenate(prediction_frag)
        self.result['label'] = np.concatenate(label_frag)
        self.result['classes'] = np.unique(self.result['label'])
        self.eval_info['mean_loss_test'] = np.mean(loss_value)
        for k in self.arg.show_topk:  # calculate top-k accuracy
            self.eval_info[f'top_{k}_accuracy'] = self.calculate_topk(k)

        self.show_eval_info()

    @staticmethod
    def get_parser(add_help=True):
        """
        Extended argument parser with options for prototype loss
        :param add_help: boolean flag to enable command line help
        :return: parser
        """
        # parameter priority: command line > config > default
        parser = super(ProcessorProtoNet, ProcessorProtoNet).get_parser(add_help=add_help)
        parser.description = 'ProtoNet Processor'

        # sampler
        parser.add_argument('--sampler', default=None, help='type of sampler')
        parser.add_argument('--train_sampler_args', action=DictAction, default=dict(),
                            help='arguments for training sampler')
        parser.add_argument('--test_sampler_args', action=DictAction, default=dict(),
                            help='arguments for test sampler')

        return parser


class PrototypeLoss(nn.Module):
    def forward(self, z_query, z_support, n_class, n_support, n_query):
        """
        Calculate prototype loss
        :param z_query: Query points
        :param z_support: Support points
        :param n_class: Number of classes
        :param n_support: Number of support points
        :param n_query: Number of query points
        :return: prototype loss
        """
        device = z_query.device
        # Calculate class-wise prototypes and determine distance to query samples
        z_proto = z_support.view(n_class, n_support, -1).mean(dim=1)
        distances = euclidean_dist(z_query, z_proto)

        # Create target: n_class x n_query x 1 with step-wise ascending labels in first dimension
        target = torch.arange(n_class).view(n_class, 1, 1).expand(n_class, n_query, 1)
        target = target.long().to(device)

        # Softmax over distances
        log_p_y = F.log_softmax(-distances, dim=1).view(n_class, n_query, n_class)
        loss = -log_p_y.gather(2, target).mean()

        return loss, distances


class KNNLoss(nn.Module):
    def forward(self, z_query, z_support, n_class, n_support, n_query):
        """
        Calculate modified version of prototype loss
        :param z_query: Query points
        :param z_support: Support points
        :param n_class: Number of classes
        :param n_support: Number of support points
        :param n_query: Number of query points
        :return: modified form of prototype loss
        """
        device = z_query.device

        distances = euclidean_dist(z_query, z_support)

        # Create target: n_class x n_query x 1 x n_query with block-wise ascending labels in first dimension
        target = torch.arange(n_class).view(n_class, 1, 1, 1).expand(n_class, n_query, 1, n_support)
        target = target.long().to(device)

        log_p_y = F.log_softmax(-distances, dim=1).view(n_class, n_query, n_class, n_support)
        loss = -log_p_y.gather(2, target).mean()

        return loss, distances
