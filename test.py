import torch
import numpy as np
import sys
import logging

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
    trainer.test()

    logger.info("\n\n\n")



if __name__ == '__main__':
    config = utils.read_json('./config.json')

    utils.setup_logger('valid', config['logger']['path'], 'test', 
                    level=logging.INFO,
                    screen=True, tofile=True)

    # import pdb; pdb.set_trace()

    config['name'] = "pixel-0.5-feature-0.5-learn-210309-200428"
    config['pretrained_models']['load'] = True
    config['pretrained_models']['G'] = "saved/pretrained_models/{}_G.pth".format(config['name'])
    config['pretrained_models']['D'] = "saved/pretrained_models/{}_D.pth".format(config['name'])
    config['pretrained_models']['FRCNN'] = "saved/pretrained_models/{}_FRCNN.pth".format(config['name'])

    main(config)
