import numpy as np

from feeder.feeder import Feeder


class FeederMeta(Feeder):
    """
    Feeder that handles also meta information of the data
    """

    def __init__(self, meta_path, **kwargs):
        """
        Initialize feeder
        :param meta_path: path to meta data
        :param kwargs: other arguments - passed to feeder.feeder
        """
        self.meta_path = meta_path
        self.meta = self.load_meta()

        super().__init__(**kwargs)

    def load_meta(self):
        """
        Load the meta data
        :return: loaded data
        """
        meta = np.load(self.meta_path)
        return meta

    def get_meta(self, item):
        """
        Return meta information - respect reduced action classes
        :return: meta information
        """
        return self.meta[self.indexes[item]]
