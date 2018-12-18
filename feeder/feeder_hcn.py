import torch
import torch.nn.functional as F

from feeder.feeder_meta import FeederMeta


class FeederHCN(FeederMeta):
    """
    Feeder for HCN training
    uses NTURGB+D format of the data
    """
    def __getitem__(self, item):
        data, label = super().__getitem__(item)

        data = self.reference_offset(data)
        data = self.scale_time(data, item)

        return data, label

    def scale_time(self, data, item, window=32):
        """
        Scales the sequence to the desired window length using interpolation
        :param data: data to scale
        :param item: index in data set - use to retrieve meta information
        :param window: window length
        :return: scaled data
        """
        C, T, J, P = data.shape  # N_channels (x, y, confidence), N_frames, N_joints, N_persons
        L = self.get_meta(item)

        data_tensor = torch.Tensor(data[:, :L, :, :])
        data_tensor = F.interpolate(data_tensor.view(1, C, L, J, P), size=(window, J, P)).view(C, window, J, P)
        data = data_tensor.numpy()

        return data

    @staticmethod
    def reference_offset(data):
        """
        Set zero into hips of the person
        :param data: data to shift
        :return: offset corrected data
        """
        offset = data[:, :, 0, 0]  # Reference point: person 0, joint 0 -> NTURGB+D
        data -= offset[:, :, None, None]
        return data
