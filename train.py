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

    trainer = trainers.Trainer(config, train_data_loader, valid_data_loader)
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

    # config['train']['pixel_sigma'] = 0.44
    # config['train']['feature_sigma'] = 2.5
    # config['train']['learned_weight'] = True
    # config['train']['intermediate_weight'] = 1
    # config['train']['intermediate_loss'] = True
    # config['name'] = 'fea-pix-learn-inter-fix'
    # main(config)

    # config['train']['pixel_sigma'] = 0.5
    # config['train']['feature_sigma'] = 0.5
    # config['train']['learned_weight'] = True
    # config['train']['intermediate_sigma'] = 0.5
    # config['train']['intermediate_loss'] = True
    # config['train']['intermediate_learned'] = True
    # config['name'] = 'fea-pix-learn-inter-fix'
    # main(config)

    # config['train']['intermediate_loss'] = False

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

    state_name = "pixel-0.01-feature-1-210307-223340"
    current_step = 11140
    # config['train']['learned_weight'] = True
    # config['train']['pixel_sigma'] = 0.435
    # config['train']['feature_sigma'] = 2.4
    config['train']['pixel_weight'] = 0.01
    config['train']['feature_weight'] = 1

    config['pretrained_models']['load'] = True
    config['pretrained_models']['G'] = "saved/pretrained_models/{}_G.pth".format(state_name)
    config['pretrained_models']['D'] = "saved/pretrained_models/{}_D.pth".format(state_name)
    config['pretrained_models']['FRCNN'] = "saved/pretrained_models/{}_FRCNN.pth".format(state_name)
    
    config['resume_state']['load'] = True
    config['resume_state']['state'] = "saved/training_state/{}-{}.state".format(state_name, current_step)
    config['train']['niter'] = 30000

    config['name'] = 'pixel-0.01-feature-1'

    main(config)