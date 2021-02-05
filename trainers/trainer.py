import logging
import torch
import math
import os
import time

import models
from utils import save_img, tensor2img, mkdir
import utils
from detection.engine import evaluate

logger = logging.getLogger('base')

class COWCTrainer:
    def __init__(self, config, data_loader, valid_data_loader=None):
        self.config = config
        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        n_gpu = torch.cuda.device_count()
        self.device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.train_size = int(math.ceil(self.data_loader.length / int(config['data_loader']['args']['batch_size'])))
        self.total_iters = int(config['train']['niter'])
        self.total_epochs = int(math.ceil(self.total_iters / self.train_size))
        self.model = models.EESN_FRCNN_GAN(config, self.device)

    def test(self):
        val_logger = logging.getLogger('valid')
        total_psnr = 0.0
        total_ssim = 0.0
        val_logger.info('######################{:^20}######################'.format(self.config['name']))
        for _, (image, targets) in enumerate(self.valid_data_loader):
            self.model.feed_data(image, targets)
            self.model.test()

            visuals = self.model.get_current_visuals()
            sr_img = tensor2img(visuals['SR'])  # uint8
            gt_img = tensor2img(visuals['GT'])  # uint8

            img_name = os.path.splitext(os.path.basename(image['LQ_path'][0]))[0]
            # img_dir = os.path.join(self.config['path']['valid_img'], img_name)
            # os.makedirs(img_dir, exist_ok=True)

            # # Save SR images for reference
            # save_img_path = os.path.join(img_dir, '{:s}_SR.png'.format(img_name))
            # save_img(sr_img, save_img_path)
            # # Save GT images for reference
            # save_img_path = os.path.join(img_dir, '{:s}_GT.png'.format(img_name))
            # save_img(gt_img, save_img_path)
            # # Save final_SR images for reference
            # save_img_path = os.path.join(img_dir, '{:s}_final_SR.png'.format(img_name))

            # calculate PSNR
            crop_size = self.config['scale']
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.
            cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
            cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
            psnr = utils.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = utils.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)

            val_logger.info('{:<25} # PSNR: {:.4e} # SSIM: {:.4e}'.format(img_name, psnr, ssim))

            total_psnr += psnr
            total_ssim += ssim

        avg_psnr = total_psnr / self.valid_data_loader.length
        avg_ssim = total_ssim / self.valid_data_loader.length

        val_logger.info('##### Validation # PSNR: {:.4e}'.format(avg_psnr))
        val_logger.info('##### Validation # SSIM: {:.4e}'.format(avg_ssim))

        # Evaluate detection result
        self.model.netG.eval()
        self.model.netFRCNN.eval()
        print('######################{:^20}######################'.format(self.config['name']))
        evaluate(self.model.netG, self.model.netFRCNN, self.valid_data_loader, self.device)
        self.model.netG.train()
        self.model.netFRCNN.train()

        

    def train(self):
        # Training logic for an epoch
        # for visualization use the following code (use batch size = 1):
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    self.data_loader.length, self.train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    self.total_epochs, self.total_iters))

        #### training
        current_step = 0
        logger.info('Start training from epoch: {:d}, iter: {:d}'.format(0, current_step))
        for epoch in range(self.total_epochs + 1):
            for _, (image, targets) in enumerate(self.data_loader):
                current_step += 1
                if current_step > self.total_iters:
                    break
                #### update learning rate
                self.model.update_learning_rate(current_step, warmup_iter=self.config['train']['warmup_iter'])

                #### training
                self.model.feed_data(image, targets)
                self.model.optimize_parameters(current_step)

                #### log
                if current_step % self.config['logger']['print_freq'] == 0:
                    logs = self.model.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                        epoch, current_step, self.model.get_current_learning_rate())
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    logger.info(message)

                # # validation
                # if self.do_validation and current_step % self.config['train']['val_freq'] == 0:
                #     self.test()

        logger.info('Saving the final model.')
        # self.model.save(utils.get_timestamp)
        self.model.save(self.config['name'])
        logger.info('End of training.')
