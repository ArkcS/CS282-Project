import torch
from torch import nn
from torch.nn import functional as F

class coarse_reconstruct(nn.Module):
    def __init__(self,in_channels,head_conv,feature_dim):
        super().__init__()

        #Voxelization: input 3d-features, output x*y*z*1 binary occupany value(0 for air, 1 for object).
        self.voxelize=nn.Sequential(
            nn.Conv1d(in_channels,head_conv,1),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_conv,1,1)
        )

        #Get sdf-feature: input 3d-features, output x*y*z*feature-dim of sdf-features.
        self.get_sdf_feat = nn.Sequential(
            nn.Conv1d(in_channels, head_conv, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_conv, feature_dim, 1)
        )

    def forward(self, features):
        Batch_size, feature_dim, X, Y, Z = features.shape
        features = features.view(Batch_size, feature_dim, X*Y*Z)

        voxel = self.voxelize(features)
        voxel = torch.sigmoid(voxel)
        voxel = voxel.permute(0, 2, 1)

        sdf_features = self.get_sdf_feat(features)
        sdf_features = sdf_features.permute(0, 2, 1)

        return voxel, sdf_features

"""input=torch.randn(8,64,40,60,20)
net=coarse_reconstruct(64,128,64)
out=net(input)
print(out)"""