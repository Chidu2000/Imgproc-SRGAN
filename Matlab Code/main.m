
% Load necessary libraries
addpath('mode');
addpath('test');
addpath('testOnly_data');


% Parse command line arguments
parser = inputParser;
addParameter(parser, 'LR_path', '../custom_dataset/DIV2K_train_LR');
addParameter(parser, 'GT_path', '../custom_dataset/DIV2K_train_HR');
addParameter(parser, 'res_num', 16);
addParameter(parser, 'num_workers', 0);
addParameter(parser, 'batch_size', 16);
addParameter(parser, 'L2_coeff', 1.0);
addParameter(parser, 'adv_coeff', 1e-3);
addParameter(parser, 'tv_loss_coeff', 0.0);
addParameter(parser, 'pre_train_epoch', 1000);
addParameter(parser, 'fine_train_epoch', 4000);
addParameter(parser, 'scale', 4);
addParameter(parser, 'patch_size', 24);
addParameter(parser, 'feat_layer', 'relu5_4');
addParameter(parser, 'vgg_rescale_coeff', 0.006);
addParameter(parser, 'fine_tuning', false);
addParameter(parser, 'in_memory', true);
addParameter(parser, 'generator_path', '');
addParameter(parser, 'mode', 'train');
parse(parser, 'LR_path', '../custom_dataset/DIV2K_train_LR', 'GT_path', '../custom_dataset/DIV2K_train_HR', 'res_num', 16, 'num_workers', 0, 'batch_size', 16, 'L2_coeff', 1.0, 'adv_coeff', 1e-3, 'tv_loss_coeff', 0.0, 'pre_train_epoch', 1000, 'fine_train_epoch', 4000, 'scale', 4, 'patch_size', 24, 'feat_layer', 'relu5_4', 'vgg_rescale_coeff', 0.006, 'fine_tuning', false, 'in_memory', true, 'generator_path', '', 'mode', 'train');

% Call the appropriate function based on the mode
if strcmp(parser.Results.mode, 'train')
    mode(parser.Results);
elseif strcmp(parser.Results.mode, 'test')
    test(parser.Results);
elseif strcmp(parser.Results.mode, 'test_only')
    test_only(parser.Results);
end

% Clear GPU memory
clear_gpu_memory();

function clear_gpu_memory()
    gpuDevice(1);
    reset(gpuDevice);
end

