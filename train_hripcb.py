import torch
import numpy as np
import sys
import logging
import os

import data
import trainers
import utils

'''
nohup stdbuf -o0 python train_hripcb.py > ./saved_hripcb/logs/log.log &
'''

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = logging.getLogger('base')
    logger.info('######################{:^20}######################'.format(config['name']))

    train_data_loader = data.COWCorHRIPCBDataLoader(
        config["data_loader"]["train"]["HR_img_dir"],
        config["data_loader"]["train"]["LR_img_dir"],
        config["data_loader"]["args"]["batch_size"], 
        config["data_loader"]["args"]["shuffle"], 
        config["data_loader"]["args"]["validation_split"], 
        config["data_loader"]["args"]["num_workers"], 
        config["data_loader"]["args"]["mean"], 
        config["data_loader"]["args"]["std"], 
        training = True)
    valid_data_loader = data.COWCorHRIPCBDataLoader(
        config["data_loader"]["valid"]["HR_img_dir"],
        config["data_loader"]["valid"]["LR_img_dir"],
        dataset_mean = config['data_loader']['args']['mean'],
        dataset_std = config['data_loader']['args']['std'],
        batch_size = 3, training = False)

    trainer = trainers.Trainer(config, train_data_loader, valid_data_loader)
    trainer.train()
    trainer.test()

    logger.info("\n\n\n")



if __name__ == '__main__':
    config = utils.read_json('./config_hripcb.json')

    utils.setup_logger('base', config['logger']['path'], 'train', 
                    level=logging.INFO,
                    screen=False, tofile=True)
    utils.setup_logger('valid', config['logger']['path'], 'valid', 
                    level=logging.INFO,
                    screen=True, tofile=True)

    config['train']['niter'] = 30000

    weights_pairs = [
                        [0.01, 1],
                        # [0.1, 1],
                        # [1, 1],
                        # [10, 1],
                    ]

    for pixel_weight, feature_weight in weights_pairs:
        config['train']['pixel_weight'] = pixel_weight
        config['train']['feature_weight'] = feature_weight

        config['train']['learned_weight'] = False
        config['name'] = 'pixel-{}-feature-{}'.format(pixel_weight, feature_weight)
        main(config)

    # config['train']['pixel_sigma'] = 0.5
    # config['train']['feature_sigma'] = 0.5
    # config['train']['learned_weight'] = True
    # config['name'] = 'pixel-{}-feature-{}-learn'.format(config['train']['pixel_sigma'], config['train']['feature_sigma'])
    # main(config)

    # config['name'] = "pixel-0.01-feature-1-210315-100912"
    # current_step = 10000
    # config['train']['pixel_weight'] = 0.01
    # config['train']['feature_weight'] = 1

    # config['pretrained_models']['load'] = True
    # config['pretrained_models']['G'] = os.path.join(config['pretrained_models']['path'], "{}_G.pth".format(config['name']))
    # config['pretrained_models']['D'] = os.path.join(config['pretrained_models']['path'], "{}_D.pth".format(config['name']))
    # config['pretrained_models']['FRCNN'] = os.path.join(config['pretrained_models']['path'], "{}_FRCNN.pth".format(config['name']))

    # config['resume_state']['load'] = True
    # config['resume_state']['state'] = os.path.join(config['resume_state']['path'], "{}-{}.state".format(config['name'], current_step))
    # config['train']['niter'] = 25000

    # main(config)