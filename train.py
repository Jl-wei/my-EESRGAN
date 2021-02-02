import torch
import numpy as np
import sys
import logging

import data
import trainers
import utils


# fix random seeds for reproducibility
SEED = 111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
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
    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    config = utils.read_json('./config.json')
    main(config)
