import torch
import numpy as np
import data
'''
python train.py -c config_GAN.json
'''

# fix random seeds for reproducibility
SEED = 111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    train_data_loader = data.COWCDataLoader(
        '/Users/jlwei/workspace/datasets/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/',
        '/Users/jlwei/workspace/datasets/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/', 
        1, training = True)
    valid_data_loader = data.COWCDataLoader(
        '/Users/jlwei/workspace/datasets/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/',
        '/Users/jlwei/workspace/datasets/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/', 
        1, training = False)

    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    config = {}
    main(config)
