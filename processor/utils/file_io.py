import pickle

import h5py
import torch

from processor.utils.utils import import_class


class FileIO:
    """
    Module to load PyTorch components and for basic file i/o
    """
    def __init__(self, logger, work_dir):
        """
        Initialize File IO
        :param logger: initialized logger - logs all loading / saving / initializing activities
        :param work_dir: path to work directory where files are saved
        """
        self.model_text = ''
        self.logger = logger
        self.work_dir = work_dir

    def load_model(self, model, **model_args):
        """
        Load PyTorch model
        :param model: class name of model
        :param model_args: arguments for model
        :return: initialized model
        """
        Model = import_class(model)
        model = Model(**model_args)
        self.model_text += '\n\n' + str(model)
        self.logger.print_log(f'Model: {Model.__name__} initialized')
        return model

    def load_weights(self, model, weights_path, ignore_weights=None):
        """
        Initialize PyTorch model with weights
        :param model: PyTorch model to initialize
        :param weights_path: path to weights
        :param ignore_weights: list of weights that should not be loaded
        :return: model with loaded weights
        """
        if ignore_weights is None:
            ignore_weights = []
        if isinstance(ignore_weights, str):
            ignore_weights = [ignore_weights]

        self.logger.print_log(f'Load weights from {weights_path}.')
        weights = torch.load(weights_path)

        # filter weights
        for i in ignore_weights:
            ignore_name = list()
            for w in weights:
                if w.find(i) == 0:
                    ignore_name.append(w)
            for n in ignore_name:
                weights.pop(n)
                self.logger.print_log(f'Filter [{i}] remove weights [{n}].')

        for w in weights:
            self.logger.print_log(f'Load weights [{w}].')

        try:
            model.load_state_dict(weights)
        except (KeyError, RuntimeError):
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            for d in diff:
                self.logger.print_log(f'Can not find weights [{d}].')
            state.update(weights)
            model.load_state_dict(state)
        return model

    def load_loss(self, loss):
        """
        Initialize PyTorch loss function
        :param loss: class name of loss
        :return: initialized loss
        """
        Loss = import_class(loss)
        loss = Loss()
        self.logger.print_log(f'Loss: {Loss.__name__} initialized')
        return loss

    def load_optimizer(self, optimizer, model, **optimizer_args):
        """
        Initialize PyTorch optimizer for model training
        :param optimizer: class name of optimizer
        :param model: model to optimize
        :param optimizer_args: arguments for optimizer
        :return: initialized optimizer
        """
        Optimizer = import_class(optimizer)
        optimizer = Optimizer(model.parameters(), **optimizer_args)
        self.logger.print_log(f'Optimizer: {Optimizer.__name__} initialized')
        return optimizer

    def load_scheduler(self, scheduler, optimizer, **scheduler_args):
        """
        Initialize PyTorch scheduler
        :param scheduler: class name of scheduler
        :param optimizer: initialized optimizer
        :param scheduler_args: arguments for scheduler
        :return: initialized scheduler
        """
        Scheduler = import_class(scheduler)
        scheduler = Scheduler(optimizer, **scheduler_args)
        self.logger.print_log(f'Scheduler: {Scheduler.__name__} initialized')
        return scheduler

    def load_feeder(self, feeder, **feeder_args):
        """
        Initialize data feeder
        :param feeder: class name of feeder
        :param feeder_args: arguments for feeder
        :return: initialized feeder
        """
        Feeder = import_class(feeder)
        feeder = Feeder(**feeder_args)
        self.logger.print_log(f'Feeder: {Feeder.__name__} initialized')
        return feeder

    def load_sampler(self, sampler, **sampler_args):
        """
        Initialize data sampler
        :param sampler: class name of sampler
        :param sampler_args: arguments for sampler
        :return: initialized sampler
        """
        Sampler = import_class(sampler)
        sampler = Sampler(**sampler_args)
        self.logger.print_log(f'Sampler: {Sampler.__name__} initialized')
        return sampler

    def save_weights(self, model, name):
        """
        Save model weights
        :param model: PyTorch model
        :param name: name of weights file
        """
        model_path = f'{self.work_dir}/{name}'
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
        weights = model.state_dict() if not isinstance(model, torch.nn.DataParallel) else model.module.state_dict()
        torch.save(weights, model_path)
        self.logger.print_log(f'The model has been saved as {model_path}.')

    def save_pkl(self, result, filename):
        """
        Save something as pickle
        :param result: result to save
        :param filename: name of pickle file
        """
        with open(f'{self.work_dir}/{filename}', 'wb') as f:
            pickle.dump(result, f)

    def save_h5(self, result, filename):
        """
        Save something as hdf5
        :param result: result to save
        :param filename: name of hdf5 file
        """
        with h5py.File(f'{self.work_dir}/{filename}', 'w') as f:
            for k in result.keys():
                f[k] = result[k]
