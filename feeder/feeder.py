import pickle

import numpy as np
from torch.utils.data import Dataset


class Feeder(Dataset):
    """
    TODO: Add doc
    """

    def __init__(self, data_path, label_path, action_classes=-1, debug=False, mmap=True):
        """
        Initialize data feeder
        :param data_path: path to data file (.npy)
        :param label_path: path to label file (.npy)
        :param action_classes: number of action classes
        :param debug: boolean flag to enter debug mode - use only first 1000 samples
        :param mmap: boolean flag to enable mmap - might be slower but way more economical with RAM
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path

        self.sample_name, self.label, self.data = self.load_data(mmap)
        self.action_classes, self.indexes = self.reduce_actions(action_classes)

        if self.debug:
            self.reduce_debug()

    def load_data(self, mmap):
        """
        Load the data (sample name, label, data)
        :param mmap: boolean flag to enable memory mapping
        :return: sample name, label, data
        """
        # load labels
        with open(self.label_path, 'rb') as f:
            sample_name, label = pickle.load(f)
            label = np.array(label)

        # load data
        data = np.load(self.data_path, mmap_mode=('r' if mmap else None))
        return sample_name, label, data

    def reduce_actions(self, action_classes):
        """
        Reduce the set of actions, e.g. to only single person actions
        :param action_classes: Labels of the action classes to keep
        :return:
            * action_classes: list of kept action classes
            * indexes: indexes corresponding to kept samples
        """
        # Get array of all selected class labels
        if isinstance(action_classes, int) and action_classes == -1:
            action_classes = np.unique(self.label)
        elif isinstance(action_classes, int) and action_classes > 0:
            action_classes = np.arange(action_classes)
        elif isinstance(action_classes, (list, tuple)):
            action_classes = np.array(action_classes)
        else:
            raise ValueError('allowed values for action_classes:\n'
                             '    * tuple/list of selected classes\n'
                             '    * integer representing the range of classes\n'
                             '    * -1 for all classes')

        # Indexes for selected action classes
        indexes = np.concatenate([np.where(self.label == action)[0] for action in action_classes])
        # Shuffle ordered by class labels indexes
        np.random.shuffle(indexes)

        return action_classes, indexes

    def reduce_debug(self):
        """
        Debug mode - use only first 1000 samples
        """
        num_debug = 1000
        self.indexes = self.indexes[:num_debug]
        self.action_classes = np.unique(self.label[self.indexes])

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        index = self.indexes[item]
        data = np.array(self.data[index])
        label = self.label[index]

        return data, label

    def get_label(self):
        """
        Return labels - respect reduced action classes
        :return: labels
        """
        return self.label[self.indexes]

    def get_data(self):
        """
        Return data - respect reduced action classes
        warning: if data is memory mapped on the hard disk the returned data will be loaded into RAM
        :return: data
        """
        return self.data[self.indexes]
