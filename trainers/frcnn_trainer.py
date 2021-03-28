import torch
import torch.nn as nn
import numpy as np
import torchvision
import os
import cv2
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from detection.engine import train_one_epoch, evaluate_base
from detection.utils import collate_fn
from detection.transforms import ToTensor, RandomHorizontalFlip, Compose
from data import FRCNNDataset

class FRCNNTrainer:
    def __init__(self, config):
        self.config = config
        n_gpu = torch.cuda.device_count()
        self.device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')

        self.num_epochs = 25

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = self.config['detector']['num_classes'] # object and background
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(self.device)
        #self.load_model(self.config['pretrained_models']['FRCNN'], self.model)

        self.data_loader, self.data_loader_test = self.data_loaders()

    def get_transform(self, train):
        # converts the image, a PIL image, into a PyTorch Tensor
        transforms = [ToTensor()]
        if train:
            transforms.append(RandomHorizontalFlip(0.5))
        return Compose(transforms)

    def data_loaders(self):
        # use our dataset and defined transformations
        dataset = FRCNNDataset(root=self.config['data_loader']['train']['LR_img_dir'],
                    image_height=64, image_width=64, transforms=self.get_transform(train=True))
        dataset_test = FRCNNDataset(root=self.config['data_loader']['valid']['LR_img_dir'],
                    image_height=64, image_width=64, transforms=self.get_transform(train=False))

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=10, shuffle=True, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=10, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        return data_loader, data_loader_test

    def train(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(self.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, self.data_loader, self.device, epoch, print_freq=50)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate_base(self.model, self.data_loader_test, device=self.device)
            if epoch % 5 == 0:
                self.save_model(self.model, 'FRCNN_LR_LR', epoch)

    def save_model(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.config['pretrained_models']['path'], save_filename)

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_model(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)
        print("model_loaded")