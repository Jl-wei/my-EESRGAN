import logging
from collections import OrderedDict
import kornia
import torch
import torchvision
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import models
import models.lr_scheduler as lr_scheduler
from models.losses import GANLoss, CharbonnierLoss
from .gan_base_model import GANBaseModel


logger = logging.getLogger('base')

# Taken from ESRGAN BASICSR repository and modified
class EESN_FRCNN_GAN(GANBaseModel):
    def __init__(self, config, device):
        super(EESN_FRCNN_GAN, self).__init__(config, device)
        self.configG = config['network_G']
        self.configD = config['network_D']
        self.configT = config['train']
        self.configO = config['optimizer']['args']
        self.configS = config['lr_scheduler']
        self.config = config
        self.device = device

        # Generator
        self.netG = models.EESNGenerator(in_nc=self.configG['in_nc'], 
                                        out_nc=self.configG['out_nc'],
                                        nf=self.configG['nf'], 
                                        nb=self.configG['nb'])
        self.netG = self.netG.to(self.device)
        self.netG = DataParallel(self.netG)

        # Descriminator
        self.netD = models.VGGDiscriminator128(in_nc=self.configD['in_nc'], 
                                            nf=self.configD['nf'])
        self.netD = self.netD.to(self.device)
        self.netD = DataParallel(self.netD)

        # FRCNN
        self.netFRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = self.config['detector']['num_classes'] # object and background
        in_features = self.netFRCNN.roi_heads.box_predictor.cls_score.in_features
        self.netFRCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.netFRCNN.to(self.device)

        # Enable training mode
        self.netG.train()
        self.netD.train()
        self.netFRCNN.train()

        # G CharbonnierLoss for final output SR and GT HR
        self.cri_charbonnier = CharbonnierLoss().to(device)

        # intermediate loss
        if self.configT['intermediate_loss']:
            self.configT['learned_weight'] = True
            if self.configT['intermediate_learned']:
                self.l_inter_w = torch.tensor(float(self.configT['intermediate_sigma']), requires_grad = True, device=device)
            else:
                self.l_inter_w = self.configT['intermediate_weight']

        # G pixel loss
        l_pix_type = self.configT['pixel_criterion']
        if l_pix_type == 'l1':
            self.cri_pix = nn.L1Loss().to(self.device)
        elif l_pix_type == 'l2':
            self.cri_pix = nn.MSELoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))

        if self.configT['learned_weight']:
            self.l_pix_w = torch.tensor(float(self.configT['pixel_sigma']), requires_grad = True, device=device)
        elif self.configT['pixel_weight'] > 0.0:
            self.l_pix_w = self.configT['pixel_weight']
        else:
            self.cri_pix = None

        # G feature loss
        l_fea_type = self.configT['feature_criterion']
        if l_fea_type == 'l1':
            self.cri_fea = nn.L1Loss().to(self.device)
        elif l_fea_type == 'l2':
            self.cri_fea = nn.MSELoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))

        if self.configT['learned_weight']:
            self.l_fea_w = torch.tensor(float(self.configT['feature_sigma']), requires_grad = True, device=device)
        elif self.configT['feature_weight'] > 0.0:
            self.l_fea_w = self.configT['feature_weight']
        else:
            self.cri_fea = None

        if self.cri_fea:  # load VGG perceptual loss
            self.netF = models.VGGFeatureExtractor(feature_layer = 34, 
                                                use_input_norm = True, 
                                                device = self.device,
                                                dataset_mean = self.config['data_loader']['args']['mean'],
                                                dataset_std = self.config['data_loader']['args']['std'])
            self.netF = self.netF.to(self.device)
            self.netF = DataParallel(self.netF)
            self.netF.eval()

        # GD gan loss
        self.cri_gan = GANLoss(self.configT['gan_type'], 1.0, 0.0).to(self.device)
        self.l_gan_w = self.configT['gan_weight']

        # D_update_ratio and D_init_iters
        self.D_update_ratio = self.configT['D_update_ratio'] if self.configT['D_update_ratio'] else 1
        self.D_init_iters = self.configT['D_init_iters'] if self.configT['D_init_iters'] else 0

        # Optimizers
        # G
        wd_G = self.configO['weight_decay_G'] if self.configO['weight_decay_G'] else 0
        optim_params = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_params.append(v)

        weight_params = []
        if self.configT['learned_weight']:
            weight_params = [self.l_pix_w, self.l_fea_w]
            if self.configT['intermediate_learned']:
                weight_params.append(self.l_inter_w)
        self.optimizer_G = torch.optim.Adam([ 
                                                {'params': optim_params},
                                                {'params': weight_params},
                                            ], lr=self.configO['lr_G'],
                                            weight_decay=wd_G,
                                            betas=(self.configO['beta1_G'], self.configO['beta2_G']))
        self.optimizers.append(self.optimizer_G)

        # D
        wd_D = self.configO['weight_decay_D'] if self.configO['weight_decay_D'] else 0
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.configO['lr_D'],
                                            weight_decay=wd_D,
                                            betas=(self.configO['beta1_D'], self.configO['beta2_D']))
        self.optimizers.append(self.optimizer_D)

        # FRCNN -- use weight decay
        FRCNN_params = [p for p in self.netFRCNN.parameters() if p.requires_grad]
        self.optimizer_FRCNN = torch.optim.SGD(FRCNN_params, lr=0.005,
                                               momentum=0.9, weight_decay=0.0005)
        self.optimizers.append(self.optimizer_FRCNN)

        # schedulers
        if self.configS['type'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, self.configS['args']['lr_steps'],
                                                     restarts=self.configS['args']['restarts'],
                                                     weights=self.configS['args']['restart_weights'],
                                                     gamma=self.configS['args']['lr_gamma'],
                                                     clear_state=False))
        elif self.configS['type'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, self.configS['args']['T_period'], eta_min=self.configS['args']['eta_min'],
                        restarts=self.configS['args']['restarts'], weights=self.configS['args']['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

        self.log_dict = OrderedDict()

        # self.print_network()  # print network
        if config['pretrained_models']['load']:
            self.load()  # load G and D if needed


    def test(self):
        self.netG.eval()
        self.netFRCNN.eval()
        with torch.no_grad():
            self.fake_H, self.final_SR, self.x_learned_lap_fake, self.x_lap, _, _ = self.netG(self.var_L)
            _, _, _, self.x_lap_HR, _, _ = self.netG(self.var_H)
        self.netG.train()
        self.netFRCNN.train()

    '''
    The main repo did not use collate_fn and image read has different flags
    and also used np.ascontiguousarray()
    Might change my code if problem happens
    '''
    def feed_data(self, image, targets):
        self.var_L = image['image_lq'].to(self.device)
        self.var_H = image['image'].to(self.device)
        input_ref = image['ref'] if 'ref' in image else image['image']
        self.var_ref = input_ref.to(self.device)
        self.targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]


    def optimize_parameters(self, step):
        self.fake_H, self.final_SR, self.x_learned_lap_fake, _, self.intermediate_in, self.intermediate_out = self.netG(self.var_L)

        # FRCNN
        for p in self.netD.parameters():
            p.requires_grad = False

        self.intermediate_img = self.final_SR
        img_count = self.intermediate_img.size()[0]
        self.intermediate_img = [self.intermediate_img[i] for i in range(img_count)]
        loss_dict = self.netFRCNN(self.intermediate_img, self.targets)
        losses = sum(loss for loss in loss_dict.values())

        self.optimizer_FRCNN.zero_grad()
        losses.backward(retain_graph=True)
        self.optimizer_FRCNN.step()


        # Generator
        for p in self.netG.parameters():
            p.requires_grad = True

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.configT['intermediate_loss']:
                if self.configT['intermediate_learned']:
                    l_g_inter = torch.square(torch.log(self.l_inter_w)) + (1 / (2 * torch.square(self.l_inter_w))) * self.cri_pix(self.intermediate_in, self.intermediate_out)
                else:
                    l_g_inter = self.l_inter_w * self.cri_pix(self.intermediate_in, self.intermediate_out)
                l_g_total += l_g_inter
            if self.cri_pix: # pixel loss
                if self.configT['learned_weight']:
                    l_g_pix = (torch.exp(self.l_pix_w) / (torch.exp(self.l_pix_w) + torch.exp(self.l_fea_w))) * self.cri_pix(self.fake_H, self.var_H)
                else:
                    l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea: # feature loss
                real_fea = self.netF(self.var_H).detach() # don't want to backpropagate this, need proper explanation
                fake_fea = self.netF(self.fake_H) # In netF normalize=False, check it
                if self.configT['learned_weight']:
                    l_g_fea = (torch.exp(self.l_fea_w) / (torch.exp(self.l_pix_w) + torch.exp(self.l_fea_w))) * self.cri_fea(fake_fea, real_fea)
                else:
                    l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea


            pred_g_fake = self.netD(self.fake_H)
            if self.configT['gan_type'] == 'gan':
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                l_g_total += l_g_gan
            elif self.configT['gan_type'] == 'ragan':
                pred_d_real = self.netD(self.var_ref).detach()
                l_g_gan = self.l_gan_w * (
                    self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                l_g_total += l_g_gan

            # EESN calculate loss
            self.lap_HR = kornia.laplacian(self.var_H, 3)
            if self.cri_charbonnier: # charbonnier pixel loss HR and SR
                l_e_charbonnier = 5 * (self.cri_charbonnier(self.final_SR, self.var_H)
                                        + self.cri_charbonnier(self.x_learned_lap_fake, self.lap_HR)) # change the weight to empirically
                l_g_total += l_e_charbonnier

            self.optimizer_G.zero_grad()
            l_g_total.backward(retain_graph=True)
            self.optimizer_G.step()


        # Discriminator
        for p in self.netD.parameters():
            p.requires_grad = True

        l_d_total = 0
        pred_d_real = self.netD(self.var_ref)
        pred_d_fake = self.netD(self.fake_H.detach()) # to avoid BP to Generator
        if self.configT['gan_type'] == 'gan':
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif self.configT['gan_type'] == 'ragan':
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2 # thinking of adding final sr d loss

        self.optimizer_D.zero_grad()
        l_d_total.backward()
        self.optimizer_D.step()


        # Log
        self.log_dict['l_Generator'] = l_g_total.item()
        self.log_dict['l_Discriminator'] = l_d_total.item()
        self.log_dict['l_FRCNN'] = losses.item()

        # if step % self.D_update_ratio == 0 and step > self.D_init_iters:
        if self.configT['intermediate_loss']:
            self.log_dict['l_g_inter'] = l_g_inter.item()
            if self.configT['intermediate_learned']:
                self.log_dict['weight_inter'] = (1 / (2 * torch.square(self.l_inter_w))).item()
        if self.cri_pix:
            self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.configT['learned_weight']:
                self.log_dict['weight_pix'] = (torch.exp(self.l_pix_w) / (torch.exp(self.l_pix_w) + torch.exp(self.l_fea_w))).item()
        if self.cri_fea:
            self.log_dict['l_g_fea'] = l_g_fea.item()
            if self.configT['learned_weight']:
                self.log_dict['weight_fea'] = (torch.exp(self.l_fea_w) / (torch.exp(self.l_pix_w) + torch.exp(self.l_fea_w))).item()
        #     self.log_dict['l_g_gan'] = l_g_gan.item()
        #     self.log_dict['l_e_charbonnier'] = l_e_charbonnier.item()

        # self.log_dict['l_d_real'] = l_d_real.item()
        # self.log_dict['l_d_fake'] = l_d_fake.item()
        # self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        # self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())


    def get_current_log(self):
        return self.log_dict


    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['lap_learned'] = self.x_learned_lap_fake.detach()[0].float().cpu()
        out_dict['lap_HR'] = self.x_lap_HR.detach()[0].float().cpu()
        out_dict['lap'] = self.x_lap.detach()[0].float().cpu()
        out_dict['final_SR'] = self.final_SR.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict


    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        # Discriminator
        s, n = self.get_network_description(self.netD)
        if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                             self.netD.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netD.__class__.__name__)

        logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        if self.cri_fea:  # F, Perceptual Network
            s, n = self.get_network_description(self.netF)
            if isinstance(self.netF, nn.DataParallel) or isinstance(
                    self.netF, DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                 self.netF.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netF.__class__.__name__)

            logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

        # FRCNN_model
        # Discriminator
        s, n = self.get_network_description(self.netFRCNN)
        if isinstance(self.netFRCNN, nn.DataParallel) or isinstance(self.netFRCNN,
                                                                DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netFRCNN.__class__.__name__,
                                             self.netFRCNN.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netFRCNN.__class__.__name__)

        logger.info('Network FRCNN structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)


    def load(self):
        load_path_G = self.config['pretrained_models']['G']
        if load_path_G:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.config['pretrained_models']['strict_load'])
        load_path_D = self.config['pretrained_models']['D']
        if load_path_D:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.config['pretrained_models']['strict_load'])
        load_path_FRCNN = self.config['pretrained_models']['FRCNN']
        if load_path_FRCNN:
            logger.info('Loading model for FRCNN [{:s}] ...'.format(load_path_FRCNN))
            self.load_network(load_path_FRCNN, self.netFRCNN, self.config['pretrained_models']['strict_load'])


    def save(self, model_name, iter_step = 0):
        if iter_step == 0:
            name = model_name
        else:
            name = "{}-{}".format(model_name, iter_step)

        self.save_network(self.netG, 'G', name)
        self.save_network(self.netD, 'D', name)
        self.save_network(self.netFRCNN, 'FRCNN', name)
