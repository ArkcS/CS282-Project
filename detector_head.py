import torch
from torch import nn
from torch.nn import functional as F


def norm_by_groups(o_channels):
    num_groups = 32
    if o_channels % 32 == 0:
        return nn.GroupNorm(num_groups, o_channels)
    else:
        return nn.GroupNorm(num_groups // 2, o_channels)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def sigmoid_hm(hm_features):
    adjusted = hm_features.sigmoid_()
    adjusted = adjusted.clamp(min=1e-4, max=1 - 1e-4)
    return adjusted


class Detector(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_classes,
            num_regression
    ):
        # num_classes is num of object categories in dataset
        # num_regression is always 8=3 + 3 + 2
        # out_channels should be 128 according to paper
        super().__init__()

        # heatmap
        self.class_head = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),

            # divide channels into groups, out_channels should be larger than 32, and divisible by 16.
            norm_by_groups(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, num_classes, kernel_size=1, padding=0, bias=True)
        )
        self.class_head[-1].bias.data.fill_(-2.19)

        # regression
        self.regression_head = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),

            # divide channels into groups, out_channels should be larger than 32, and divisible by 16.
            norm_by_groups(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, num_regression, kernel_size=1, padding=0, bias=True)
        )
        fill_fc_weights(self.regression_head)

    def forward(self, features):

        head_class = self.class_head(features)
        head_regression = self.regression_head(features)

        # likelihood of center
        head_class = sigmoid_hm(head_class)
        # three dimension center offset
        offset_p = head_regression[:, :3, ...].clone()
        head_regression[:, :3, ...] = torch.tanh(offset_p)
        # three dimension object size
        obj_size = head_regression[:, 3:6, ...].clone()
        head_regression[:, 3:6, ...] = torch.sigmoid(obj_size) - 0.5
        # 2 dimension rotation angle
        angles = head_regression[:, 6:, ...].clone()
        head_regression[:, 6:, ...] = F.normalize(angles)

        return head_class, head_regression


if __name__ == '__main__':
    a = torch.randn(1, 64, 20, 20, 20)
    # batch_size num_channels positions[3]
    m = Detector(64, 128, 8, 8)
    b, c = m.forward(a)
    print(b.shape, c.shape)
    print('end')