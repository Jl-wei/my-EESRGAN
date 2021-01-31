import logging
import torch
import math
import os
import time

import models
from utils import save_img, tensor2img, mkdir

logger = logging.getLogger('base')

class COWCTrainer:
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

        n_gpu = torch.cuda.device_count()
        self.device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.train_size = int(math.ceil(self.data_loader.length / int(config['data_loader']['args']['batch_size'])))
        self.total_iters = int(config['train']['niter'])
        self.total_epochs = int(math.ceil(self.total_iters / self.train_size))
        self.model = models.EESN_FRCNN_GAN(config, self.device)

    def train(self):
        # Training logic for an epoch
        # for visualization use the following code (use batch size = 1):
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    self.data_loader.length, self.train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    self.total_epochs, self.total_iters))

        # #### training
        # current_step = 0
        # logger.info('Start training from epoch: {:d}, iter: {:d}'.format(0, current_step))
        # for epoch in range(self.total_epochs + 1):
        #     for _, (image, targets) in enumerate(self.data_loader):
        #         current_step += 1
        #         if current_step > self.total_iters:
        #             break
        #         #### update learning rate
        #         self.model.update_learning_rate(current_step, warmup_iter=self.config['train']['warmup_iter'])

        #         #### training
        #         self.model.feed_data(image, targets)
        #         self.model.optimize_parameters(current_step)

        #         #### log
        #         if current_step % self.config['logger']['print_freq'] == 0:
        #             logs = self.model.get_current_log()
        #             message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
        #                 epoch, current_step, self.model.get_current_learning_rate())
        #             for k, v in logs.items():
        #                 message += '{:s}: {:.4e} '.format(k, v)
        #             logger.info(message)

        logger.info('Saving the final model.')
        self.model.save(time.strftime("%Y%m%d%H%M%S"))
        logger.info('End of training.')
