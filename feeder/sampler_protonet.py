import numpy as np
from torch.utils.data import Sampler


class SamplerProtoNet(Sampler):
    """
    Sampler for the training of the prototype loss
    """
    def __init__(self, labels, num_classes, num_support, num_query, num_episodes):
        """
        Initialize the prototype sampler
        :param labels: labels of the data set
        :param num_classes: Number of action classes
        :param num_support: Number of support samples
        :param num_query: Number of query samples
        :param num_episodes: Number of episodes per epoch
        """
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.num_samples = num_support + num_query
        self.num_episodes = num_episodes

        self.action_classes = np.unique(labels)
        self.data_indexes = {}
        for i in self.action_classes:
            self.data_indexes[i] = np.argwhere(labels == i).reshape(-1)

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        """
        Sampling support and query samples for one epoch
        :return: batch of indexes [num_support + num_query]
        """
        for it in range(self.num_episodes):
            # Sample N_classes unique class indexes
            sampled_classes = np.random.choice(self.action_classes, size=self.num_classes, replace=False)

            batch = []
            for c in sampled_classes:
                # Sample (N_support + N_query == N_sample) unique data indexes
                class_indexes = self.data_indexes[c]
                # Permute in-class order and get corresponding class indexes
                batch.append(class_indexes[np.random.permutation(len(class_indexes))[:self.num_samples]])
            batch = np.stack(batch)
            # Re-order indexes to match [indexes_support, indexes_query]
            batch = np.concatenate([batch[:, :self.num_support].reshape(-1), batch[:, self.num_support:].reshape(-1)])
            yield batch
