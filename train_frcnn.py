import torch
import numpy as np
import sys
import logging
import os

import data
import trainers
import utils

'''
nohup stdbuf -o0 python train_frcnn.py > ./saved_hripcb/logs/frcnn_log.log &
'''

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


if __name__ == '__main__':
    config = utils.read_json('./config_hripcb.json')

    # config['pretrained_models']['load'] = True
    # config['pretrained_models']['FRCNN'] = 'saved_hripcb/pretrained_models/10_FRCNN_HR_HR.pth'
    trainer = trainers.FRCNNTrainer(config)
    trainer.train()
    # trainer.test()

