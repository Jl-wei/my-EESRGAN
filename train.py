import torch
import numpy as np
import sys
import logging

import data
import trainers
import utils

'''
stdbuf -oL python train.py > ./saved/logs/log
'''

# fix random seeds for reproducibility
SEED = 111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    utils.setup_logger('base', config['logger']['path'], 'train_' + config['name'], 
                    level=logging.INFO,
                    screen=True, tofile=True)
    utils.setup_logger('valid', config['logger']['path'], 'valid_' + config['name'], 
                    level=logging.INFO,
                    screen=True, tofile=True)

    logger = logging.getLogger('base')
    logger.info('######################{:^20}######################'.format(config['name']))

    train_data_loader = data.COWCDataLoader(
        config["data_loader"]["train"]["HR_img_dir"],
        config["data_loader"]["train"]["LR_img_dir"],
        config["data_loader"]["args"]["batch_size"], 
        config["data_loader"]["args"]["shuffle"], 
        config["data_loader"]["args"]["validation_split"], 
        config["data_loader"]["args"]["num_workers"], 
        training = True)
    valid_data_loader = data.COWCDataLoader(
        config["data_loader"]["valid"]["HR_img_dir"],
        config["data_loader"]["valid"]["LR_img_dir"],
        1, training = False)

    trainer = trainers.COWCTrainer(config, train_data_loader, valid_data_loader)
    trainer.train()
    trainer.test()
    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    config = utils.read_json('./config.json')

    config['name'] = 'pixel-001-feature-1'
    main(config)

    config['name'] = 'pixel-001-feature-10'
    # config['train']['pixel_weight'] = 0.01
    config['train']['feature_weight'] = 10
    main(config)

    config['name'] = 'pixel-001-feature-100'
    config['train']['feature_weight'] = 100
    main(config)

    config['name'] = 'pixel-001-feature-1000'
    config['train']['feature_weight'] = 1000
    main(config)

    config['name'] = 'pixel-001-feature-10000'
    config['train']['feature_weight'] = 10000
    main(config)
