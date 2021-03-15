import torch
import numpy as np
import sys
import logging
import os

import data
import trainers
import utils

'''
nohup stdbuf -oL python test.py > ./saved/logs/test.log &
'''

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = logging.getLogger('valid')
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
        batch_size=1, training = False)

    trainer = trainers.Trainer(config, train_data_loader, valid_data_loader)
    trainer.test()

    logger.info("\n\n\n")



if __name__ == '__main__':
    config = utils.read_json('./config.json')

    utils.setup_logger('valid', config['logger']['path'], 'test', 
                    level=logging.INFO,
                    screen=True, tofile=True)

    # import pdb; pdb.set_trace()

    config['test']['save_img'] = True

    config['name'] = "pixel-0.01-feature-1-210307-223340"
    config['pretrained_models']['load'] = True
    config['pretrained_models']['G'] = os.path.join(config['pretrained_models']['path'], "{}_G.pth".format(config['name']))
    config['pretrained_models']['D'] = os.path.join(config['pretrained_models']['path'], "{}_D.pth".format(config['name']))
    config['pretrained_models']['FRCNN'] = os.path.join(config['pretrained_models']['path'], "{}_FRCNN.pth".format(config['name']))

    main(config)
