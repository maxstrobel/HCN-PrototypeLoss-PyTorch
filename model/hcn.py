import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    """
    Standard module's weight initialization
    :param m: pytorch module
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    """
    Hierarchical Co-occurrence Network
    Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation
    https://arxiv.org/abs/1804.06055
    """

    def __init__(self, in_channels, num_class, window_size=32, out_channels=64, num_joint=25, num_person=2):
        """
        Initializer function for HCN
        :param in_channels: Number of in channels, e.g. 3 for (x,y,confidence)
        :param num_class: Number of action classes
        :param window_size: Number of frames per action
        :param out_channels: Intermediate channel size parameter
        :param num_joint: Number of joints per person
        :param num_person: Number of persons per frame
        """
        super().__init__()
        self.__name__ = 'HCN'
        self.num_person = num_person

        self.conv_position = ConvBlockBase(in_channels, out_channels, window_size, num_joint)
        self.conv_motion = ConvBlockBase(in_channels, out_channels, window_size, num_joint)

        self.conv_cooccurrence = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear((out_channels * 4) * (window_size // 16) ** 2, 512),  # 4*4 for window=64; 8*8 for window=128
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(512, num_class)
        )

        self.apply(weights_init)

    def forward(self, x):
        N, C, T, J, P = x.size()  # N_samples, N_channels (x, y, confidence), N_frames, N_joints, N_persons

        x_mot = F.interpolate(x[:, :, 1::, :, :] - x[:, :, 0:-1, :, :], size=(T, J, P))

        # Extract features for all persons
        features = []
        for p in range(self.num_person):
            out_position = self.conv_position(x[:, :, :, :, p])
            out_motion = self.conv_motion(x_mot[:, :, :, :, p])

            out = torch.cat((out_position, out_motion), dim=1)
            out = self.conv_cooccurrence(out)
            features.append(out)

        # Merge multiple persons - available otions max/mean/concat (sec. 3.4)
        out = torch.max(*features).view(N, -1)
        out = self.classifier(out)

        return out


class ConvBlockBase(nn.Module):
    """
    Convolution block of HCN
    """

    def __init__(self, in_channels, out_channels, window_size, num_joint):
        """
        :param in_channels: Number of in channels - usually 3 for (x,y,z)
        :param window_size: Number of frames per action
        :param out_channels: Intermediate channel size parameter
        :param num_joint: Number of joints per person
        """
        super().__init__()

        self.conv_block_base = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=window_size, kernel_size=(3, 1), padding=(1, 0))
        )

        self.conv_block_center = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channels // 2, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.5, inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv_block_base(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.conv_block_center(x)
        return x
