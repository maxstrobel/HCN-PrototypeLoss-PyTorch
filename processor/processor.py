import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from processor.utils.argument_parser import ArgumentParser
from processor.utils.file_io import FileIO
from processor.utils.logger import Logger
from processor.utils.utils import DictAction


class Processor(ArgumentParser):
    """
    Inspired by https://github.com/yysijie/st-gcn
    Processor that handles
        * Training of the model
        * Testing / evaluation of the model
        * Initializations of the whole training procedure
        * Logging
        * Load & save results / models
    """

    def __init__(self, argv):
        """
        Load the configuration from command line and a specified config file
        Initialize logging, file i/o, model, training environment
        :param argv: arguments from command line
        """
        super().__init__(argv)
        self.logger = Logger(self.work_dir, self.arg.save_log, self.arg.print_log)
        self.fileio = FileIO(self.logger, self.work_dir)

        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.eval_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

        self.model = self.load_model()
        self.load_weights()
        self.loss = self.load_loss()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()
        self.data_loader = self.load_data()
        self.dev = self.device()

        self.logger.print_log('Processor: Environment initialized')

    def load_model(self):
        """
        Load & initialize model specified in configuration
        :return: initialized model
        """
        model = self.fileio.load_model(self.arg.model, **self.arg.model_args)
        dummy_input = torch.Tensor(1, 3, 32, 25, 2)   # Dummy input to generate a network graph in tensorboard
        self.logger.writer.add_graph(model, (dummy_input,))
        return model

    def load_weights(self):
        """
        Load specified weights into model
        """
        if self.arg.weights:
            self.fileio.load_weights(self.model, self.arg.weights, self.arg.ignore_weights)

    def load_loss(self):
        """
        Load specified loss
        :return: loss
        """
        if self.arg.loss:
            loss = self.fileio.load_loss(self.arg.loss)
        else:
            loss = None
        return loss

    def load_optimizer(self):
        """
        Load specified optimizer
        :return: optimizer
        """
        return self.fileio.load_optimizer(self.arg.optimizer, self.model, **self.arg.optimizer_args)

    def load_scheduler(self):
        """
        Load specified scheduler
        :return: scheduler
        """
        if self.arg.scheduler:
            scheduler = self.fileio.load_scheduler(self.arg.scheduler, self.optimizer, **self.arg.scheduler_args)
        else:
            scheduler = None
        return scheduler

    def load_data(self):
        """
        Load data and use specified data feeder and data sampler
        :return:
        """
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        if 'debug' not in self.arg.test_feeder_args:
            self.arg.test_feeder_args['debug'] = self.arg.debug

        data_loader = dict()
        if self.arg.phase == 'train':
            data_loader['train'] = DataLoader(
                dataset=self.fileio.load_feeder(self.arg.feeder, **self.arg.train_feeder_args),
                batch_size=self.arg.train_batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True)
            self.logger.print_log(f'DataLoader: {len(data_loader["train"].dataset)} training samples loaded')
        if self.arg.test_feeder_args:
            data_loader['test'] = DataLoader(
                dataset=self.fileio.load_feeder(self.arg.feeder, **self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker)
            self.logger.print_log(f'DataLoader: {len(data_loader["test"].dataset)} test samples loaded')

        return data_loader

    def device(self):
        """
        Set used device: CPU / single GPU
        :return: used devuce
        """
        if self.arg.use_gpu and torch.cuda.device_count():
            dev = "cuda:0"  # single GPU
        else:
            dev = "cpu"

        # move modules to selected device
        self.model = self.model.to(dev)
        return dev

    def start(self):
        """
        Start training of an model
        This function unifies the whole procedure on a very high level with
            * model training
            * model evaluation
            * saving weights
            * parametrization of optimizer / scheduler
            * logging
        """
        self.logger.print_log(f'Parameters:\n{str(vars(self.arg))}\n')

        # training phase
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.logger.print_log(f'Training epoch: {epoch}')
                self.train()

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    filename = f'epoch{epoch + 1}_model.pt'
                    self.fileio.save_weights(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    self.logger.print_log(f'Eval epoch: {epoch}')
                    self.test()

                # scheduler
                if self.scheduler:
                    self.scheduler.step()

        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.logger.print_log(f'Model:   {self.arg.model}.')
            self.logger.print_log(f'Weights: {self.arg.weights}.')

            # evaluation
            self.logger.print_log('Evaluation Start:')
            self.test()

    def train(self):
        """
        Train model an epoch
        This function is the real training of the model with
            * forward pass
            * backward pass
            * optimization of weights
            * logging of single iterations
        """
        self.model.train()
        loader = self.data_loader['train']
        loss_value = []

        with tqdm(total=len(loader)) as t:
            for data, label in loader:
                # get data
                data = data.float().to(self.dev)
                label = label.long().to(self.dev)

                # forward
                output = self.model(data)
                loss = self.loss(output, label)

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
        Test model and print out / store statistics
        """
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        with tqdm(total=len(loader)) as t:
            for data, label in loader:
                # get data
                data = data.float().to(self.dev)
                label = label.long().to(self.dev)

                # inference
                with torch.no_grad():
                    output = self.model(data)
                    loss = self.loss(output, label)

                # statistics
                result_frag.append(output.data.cpu().numpy())
                loss_value.append(loss)
                label_frag.append(label.data.cpu().numpy())

                self.iter_info['loss'] = loss.data.item()
                self.update_progress_bar(t)

        # evaluation
        self.result['output'] = np.concatenate(result_frag)
        self.result['prediction'] = self.result['output'].argsort()[:, ::-1]  # sort with descending probabilities
        self.result['label'] = np.concatenate(label_frag)
        self.result['classes'] = np.unique(self.result['label'])
        self.eval_info['mean_loss_test'] = np.mean(loss_value)
        for k in self.arg.show_topk:  # calculate top-k accuracy
            self.eval_info[f'top_{k}_accuracy'] = self.calculate_topk(k)

        self.show_eval_info()

    def show_epoch_info(self):
        """
        Show informations per epoch
        """
        for k, v in self.epoch_info.items():
            self.logger.print_log(f'EpochInfo\t{k}: {v}')
        self.logger.tensorboard_log(self.epoch_info, self.meta_info['epoch'])

    def show_eval_info(self):
        """
        Show extended informations after an evaluation / testing phase
        """
        for k, v in self.eval_info.items():
            self.logger.print_log(f'EvalInfo\t{k}: {v}')
        self.logger.tensorboard_log(self.eval_info, self.meta_info['epoch'])
        self.logger.tensorboard_confusion_matrix(ground_truth=self.result['label'],
                                                 prediction=self.result['prediction'][:, 0],
                                                 classes=self.result['classes'],
                                                 global_step=self.meta_info['epoch'])
        # Random sample N embeddings for projection visualization
        num_embeddings = 500
        indexes_embeddings = np.random.permutation(self.result['label'].shape[0])[:num_embeddings]
        self.logger.tensorboard_embedding(self.result['output'][indexes_embeddings],
                                          self.result['label'][indexes_embeddings],
                                          self.meta_info['epoch'])

    def show_iter_info(self, progress_bar):
        """
        Show informations per iteration
        :param progress_bar: tqdm progress bar
        """
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            self.logger.tensorboard_log(self.iter_info, self.meta_info['iter'])
        self.update_progress_bar(progress_bar)

    def update_progress_bar(self, progress_bar):
        """
        Update progress bar during training / testing
        :param progress_bar: tqdm progress bar
        """
        progress_bar.set_postfix(loss=f'{self.iter_info["loss"]:05.3f}')
        progress_bar.update()

    def calculate_topk(self, k):
        """
        Calculate top-k accuracy
        :param k: k
        :return: accuracy
        """
        # compare label against top-k (=highest k) probabilities
        hit_top_k = [l in self.result['prediction'][i, :k] for i, l in enumerate(self.result['label'])]
        accuracy = sum(hit_top_k) * 100.0 / len(hit_top_k)
        return accuracy

    @staticmethod
    def get_parser(add_help=True):
        """
        Extended argument parser with general options for the processor
        :param add_help: boolean flag to enable command line help
        :return: parser
        """
        # parameter priority: command line > config > default
        # https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        parser = super(Processor, Processor).get_parser()
        parser.description = 'Processor'
        # parser = argparse.ArgumentParser(parents=[parent_parser], description='Processor')
        parser.add_argument('--use_gpu', action='store_true', default=False, help='use GPUs or not')
        parser.add_argument('--debug', action="store_true", default=False, help='less data, faster loading')

        # processor
        parser.add_argument('--phase', default='train', help='train or test')
        parser.add_argument('--save_result', action="store_true", default=False, help='save output of model')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='indexes of GPUs for training or testing')
        # visualize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5, help='interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', action="store_true", default=True, help='save logging or not')
        parser.add_argument('--print_log', action="store_true", default=True, help='print logging or not')
        parser.add_argument('--show_topk', type=int, default=[1, 2, 5], nargs='+', help='show top-k accuracies')
        # model
        parser.add_argument('--model', default=None, help='type of model')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='arguments for model')
        parser.add_argument('--weights', default=None, help='weights for model initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                            help='ignored weights during initialization')
        parser.add_argument('--loss', default=None, help='type of loss function')
        # optimizer
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--optimizer_args', action=DictAction, default=dict(), help='arguments for optimizer')
        # scheduler
        parser.add_argument('--scheduler', default=None, help='type of scheduler')
        parser.add_argument('--scheduler_args', action=DictAction, default=dict(), help='arguments for scheduler')
        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='type of data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(),
                            help='arguments for training data loader')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(),
                            help='arguments for test data loader')
        parser.add_argument('--train_batch_size', type=int, default=256, help='batch size for training')
        parser.add_argument('--test_batch_size', type=int, default=256, help='batch size for test')
        parser.add_argument('--num_worker', type=int, default=4, help='number of workers per gpu for data loader')

        return parser
