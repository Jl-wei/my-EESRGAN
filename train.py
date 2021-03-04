import torch
import numpy as np
import sys
import logging

import data
import trainers
import utils

'''
nohup stdbuf -oL python train.py > ./saved/logs/log.log &
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

    logger.info("\n\n\n")



if __name__ == '__main__':
    config = utils.read_json('./config.json')

    utils.setup_logger('base', config['logger']['path'], 'train', 
                    level=logging.INFO,
                    screen=True, tofile=True)
    utils.setup_logger('valid', config['logger']['path'], 'valid', 
                    level=logging.INFO,
                    screen=True, tofile=True)

    # import pdb; pdb.set_trace()

    # config['train']['pixel_sigma'] = 0.5
    # config['train']['feature_sigma'] = 0.5
    # config['train']['learned_weight'] = True
    # config['name'] = 'pixel-{}-feature-{}-learn'.format(config['train']['pixel_sigma'], config['train']['feature_sigma'])
    # main(config)

    config['train']['pixel_sigma'] = 0.45
    config['train']['feature_sigma'] = 1.58
    config['train']['learned_weight'] = True
    config['train']['intermediate_weight'] = 1
    config['train']['intermediate_loss'] = True
    config['name'] = 'fea-pix-learn-inter-fix'
    main(config)

    config['train']['intermediate_loss'] = False

    # weights_pairs = [
    #                     [10, 1],
    #                     [1, 1],
    #                     [0.1, 1],
    #                     [0.01, 1],
    #                 ]

    # for pixel_weight, feature_weight in weights_pairs:
    #     config['train']['pixel_weight'] = pixel_weight
    #     config['train']['feature_weight'] = feature_weight

    #     config['train']['learned_weight'] = False
    #     config['name'] = 'pixel-{}-feature-{}'.format(pixel_weight, feature_weight)
    #     main(config)
