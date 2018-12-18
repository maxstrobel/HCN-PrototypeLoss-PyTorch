import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter


class Logger:
    """
    Logger for all kinds of activities
    """
    def __init__(self, work_dir, save_log=True, print_log=True):
        """
        :param work_dir: path to working directory where the log is stored
        :param save_log: boolean flag to enable a log file on hard disk
        :param print_log: boolean flag to enable live output on command line
        """
        self.work_dir = work_dir
        self.save_log = save_log
        self.print_to_screen = print_log
        self.writer = SummaryWriter(log_dir=f'{self.work_dir}/tb_logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}')

    def print_log(self, log_str, print_time=True):
        """
        Log a string
        :param log_str: string to log
        :param print_time: boolean flag to enable a time stamp before the logged string
        """
        if print_time:
            log_str = time.strftime("[%d.%m.%y|%X] ", time.localtime()) + log_str

        if self.print_to_screen:
            print(log_str)
        if self.save_log:
            with open(f'{self.work_dir}/log.txt', 'a') as f:
                f.write(log_str + '\n')

    def tensorboard_log(self, info, global_step):
        """
        Log at tensorboard
        :param info: info to log
        :param global_step: step in training
        """
        for k, v in info.items():
            self.writer.add_scalar(k, v, global_step)

    def tensorboard_embedding(self, data, label, global_step):
        """
        Log data embedding in tensorboard
        :param data: data to embed
        :param label: labels corresponding to data
        :param global_step: step in training
        """
        self.writer.add_embedding(data, metadata=label, global_step=global_step)

    def tensorboard_confusion_matrix(self, ground_truth, prediction, classes, global_step):
        """
        Log a confusion matrix in tensorboard
        :param ground_truth: ground truth labels
        :param prediction: predicted labels
        :param classes: class names corresponding to label indexes
        :param global_step: step in training
        """
        fig = self.plot_confusion_matrix(ground_truth, prediction, classes)
        self.writer.add_figure('matplotlib', fig, global_step=global_step)

    @staticmethod
    def plot_confusion_matrix(ground_truth, prediction, classes):
        """
        Create a plot of a confusion matrix
        :param ground_truth: ground truth labels
        :param prediction: predicted labels
        :param classes: class names corresponding to label indexes
        :return: plot of confusion matrix
        """
        num_classes = len(classes)

        # Normalized confusion matrix
        cnf_matrix = confusion_matrix(ground_truth, prediction)
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        # Plot
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)
        cnf_plot = ax.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        fig.colorbar(cnf_plot)

        # Ticks - class names
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, classes, rotation=-70, fontsize=4)
        plt.yticks(tick_marks, fontsize=4)

        return fig
