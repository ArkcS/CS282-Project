START_EPOCH = 0
END_EPOCH = 100
SAVE_EPOCH = 5
LR = 1.0e-4


Train_dir = "./datasets/chair_train_local/"



################constant of pipeline one ##############
voxel_size = 0.1
grid_size = [4, 2.5, 2.5]
grid_offset = [-2, -2, -0.15]
num_levels = 3
d_in = 64

################constant of pipeline two #############
in_channels = 64
out_channels = 128
num_classes = 1
num_regression = 8