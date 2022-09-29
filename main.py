import os
from datetime import datetime
import sys
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np  
import scipy.io
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from config_parameter_3 import *
from tqdm import *
import cv2
from models.resnet import *
import featlift 
from detector_head import *
from coarse_reconstruct import *
from loss import *

### The dataset of images is saved in config_parameter.py
pic_path = Train_dir +"high"  #using the real pictured photo as image input
K_path = Train_dir +"camera_transform"  #The extrinsic path
RT_path = Train_dir +"view_transform"  #The intrinsic path
all_pic_num = len(os.listdir(pic_path))   #get the length of the one dataset
pca_model = "./pca_normal.mat"


#############functions ##############
def feature_lifting(input_width,input_height,voxel_size, grid_size, grid_offset, num_levels, d_in, feat8, feat16, feat32):
    # 1. compute grid_pos
    def make_grid(grid_size, grid_res, grid_offset):
        width, height, depth = grid_size
        x_grid_res, y_grid_res, z_grid_res = grid_res
        xoff, yoff, zoff = grid_offset

        xcoords = torch.arange(0., width, x_grid_res) + xoff
        ycoords = torch.arange(0., height, y_grid_res) + yoff
        zcoords = torch.arange(0., depth, z_grid_res) + zoff

        zz, yy, xx = torch.meshgrid(zcoords, ycoords, xcoords)
        grid = torch.stack([xx, yy, zz], dim=-1)
        grid = grid.permute(2,1,0,3)
        voxel_corners = torch.cat((grid[:-1, :-1, :-1].unsqueeze(-1),
                                grid[:-1, :-1, 1:].unsqueeze(-1),
                                grid[:-1, 1:, :-1].unsqueeze(-1),
                                grid[:-1, 1:, 1:].unsqueeze(-1),
                                grid[1:, :-1, :-1].unsqueeze(-1),
                                grid[1:, :-1, 1:].unsqueeze(-1),
                                grid[1:, 1:, :-1].unsqueeze(-1),
                                grid[1:, 1:, 1:].unsqueeze(-1)), dim=-1)  
        voxel_pos = torch.mean(voxel_corners, dim=-1)
        return voxel_pos

    grid_res = (voxel_size, voxel_size, voxel_size)
    grid_range = (grid_size[0]+voxel_size, grid_size[1]+voxel_size, grid_size[2]+voxel_size)
    grid_pos = make_grid(grid_range, grid_res, grid_offset)

    # 2. lifting
    lifting_network = featlift.featlifting(grid_pos, input_width, input_height, num_levels, d_in)
    feat_3d, visible = lifting_network.forward(K, RT, feat8, feat16, feat32)

    return feat_3d, visible
########################


###init of model###
backbone = resnet34(pretrained=True)
detector = Detector(in_channels, out_channels, num_classes, num_regression)
recon_network = coarse_reconstruct(64,128,64)
lr = LR
optimizer = torch.optim.Adam([backbone.parameters(),detector.parameters(),recon_network.parameters()], lr=lr)

print("Start for Training")
for epoch in tqdm(range(START_EPOCH, END_EPOCH)):     
    for data_index in range(all_pic_num):   
        ## one round input ##
        image = cv2.imread(pic_path+"/{:0>5d}.png".format(data_index))
        K = np.load(K_path+"/{:0>5d}.npy".format(data_index))
        RT = np.load(RT_path+"/{:0>5d}.npy".format(data_index))
        
        image = torch.tensor(image).unsqueeze(0).permute(0,3,1,2).float()
        K = torch.tensor(K).unsqueeze(0);RT = torch.tensor(RT).unsqueeze(0)
        input_width = image.shape[0];input_height = image.shape[1]
        #shape 1 ,3 , H, W
        feat8, feat16, feat32 = backbone.forward(image)
        #from 2D dim feature to 3D dim feature
        feat_3d, visible = feature_lifting(input_width,input_height,voxel_size, grid_size, grid_offset, num_levels, d_in, feat8, feat16, feat32)
        #predict the class and the prbobility
        predict_class, predict_regression = detector.forward(feat_3d)
        #predict the coarse and the fine of the object
        predict_voxel, predict_sdf_feat = recon_network.forward(feat_3d)
        #get all the result for loss.
        output = [feat_3d,predict_class,predict_regression,predict_voxel,predict_sdf_feat]
        each_loss = get_loss(output,pca_model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #for each epoch, show the loss 
    print('Epoch: {}, Epoch loss = {}'.format(epoch,loss))

    if epoch % SAVE_EPOCH==0:
        torch.save({"epoch": epoch, 
        "backbone_model_state_dict": backbone.state_dict(),
        "detector_model_state_dict": detector.state_dict(),
        "recon_network_model_state_dict": recon_network.state_dict()},
        "./checkpoints/{:0>5d}.pth".format(epoch)
        )

#####using the model in the wild #########  todo. A better visualization
load_epoch_name = 5
if Test:
    saved_model_state = torch.load("./checkpoints/{:0>5d}.pth".format(load_epoch_name))
    backbone.load_state_dict(saved_model_state["backbone_model_state_dict"])
    detector.load_state_dict(saved_model_state["detector_model_state_dict"])
    recon_network.load_state_dict(saved_model_state["recon_network_model_state_dict"])

    image = cv2.imread(pic_path+"/{:0>5d}.png".format(data_index))
    K = np.load(K_path+"/{:0>5d}.npy".format(data_index))
    RT = np.load(RT_path+"/{:0>5d}.npy".format(data_index))
    image = torch.tensor(image).unsqueeze(0).permute(0,3,1,2).float()
    K = torch.tensor(K).unsqueeze(0);RT = torch.tensor(RT).unsqueeze(0)
    feat8, feat16, feat32 = backbone.forward(image)
    feat_3d, visible = feature_lifting(input_width,input_height,voxel_size, grid_size, grid_offset, num_levels, d_in, feat8, feat16, feat32)
    predict_class, predict_regression = detector.forward(feat_3d)
    predict_voxel, predict_sdf_feat = recon_network.forward(feat_3d)