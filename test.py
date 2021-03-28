import torch
import numpy as np
import sys
import logging
import os
import shutil

import data
import trainers
import utils

'''
nohup stdbuf -oL python test.py config_hripcb.json > ./saved_hripcb/logs/test.log &
nohup stdbuf -oL python test.py config.json > ./saved/logs/test.log &
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
        training = False)
    valid_data_loader = data.COWCorHRIPCBDataLoader(
        config["data_loader"]["valid"]["HR_img_dir"],
        config["data_loader"]["valid"]["LR_img_dir"],
        dataset_mean = config['data_loader']['args']['mean'],
        dataset_std = config['data_loader']['args']['std'],
        batch_size = 1, training = False)

    trainer = trainers.Trainer(config, train_data_loader, valid_data_loader)
    trainer.test()

    logger.info("\n\n\n")



if __name__ == '__main__':
    config_path = sys.argv[1]
    config = utils.read_json(config_path)

    utils.setup_logger('valid', config['logger']['path'], 'test', 
                    level=logging.INFO,
                    screen=True, tofile=True)

    # import pdb; pdb.set_trace()

    config['test']['save_img'] = True
    shutil.rmtree(config['path']['valid_img'], ignore_errors=True)

    config['test']['test_frcnn'] = False
    config['test']['test_similarity'] = True

    config['name'] = "7cls-pixel-0.5-feature-0.5-learn-210318-183828-30000"
    config['pretrained_models']['load'] = True
    config['pretrained_models']['G'] = os.path.join(config['pretrained_models']['path'], "{}_G.pth".format(config['name']))
    config['pretrained_models']['D'] = os.path.join(config['pretrained_models']['path'], "{}_D.pth".format(config['name']))
    config['pretrained_models']['FRCNN'] = os.path.join(config['pretrained_models']['path'], "{}_FRCNN.pth".format(config['name']))

    # config['name'] = "pixel-0.99-feature-0.01-210301-104831"
    # config['pretrained_models']['load'] = True
    # config['pretrained_models']['G'] = os.path.join("saved/pretrained_models", "{}_G.pth".format(config['name']))
    # config['pretrained_models']['D'] = os.path.join("saved/pretrained_models", "{}_D.pth".format(config['name']))
    # config['pretrained_models']['FRCNN'] = os.path.join("saved/pretrained_models", "{}_FRCNN.pth".format(config['name']))

    main(config)
