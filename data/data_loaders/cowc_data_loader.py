import os
import sys
import torch
import albumentations as A
from .base_data_loader import BaseDataLoader
import data

class COWCDataLoader(BaseDataLoader):
    """
    COWC data loading using BaseDataLoader
    """
    def __init__(self, data_dir_GT, data_dir_LQ, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        #data transformation
        #According to this link: https://discuss.pytorch.org/t/normalization-of-input-image/34814/8
        #satellite image 0.5 is good otherwise calculate mean and std for the whole dataset.
        #calculted mean and std using method from util
        '''
        Data transform for GAN training
        '''
        data_transforms_train = A.Compose([
            A.HorizontalFlip(),
            A.Normalize( #mean std for potsdam dataset from COWC [Calculate also for spot6]
                mean=[0.3442, 0.3708, 0.3476],
                std=[0.1232, 0.1230, 0.1284]
                )
        ],
            additional_targets={
             'image_lq':'image'
            },
            bbox_params=A.BboxParams(
             format='pascal_voc',
             min_area=0,
             min_visibility=0,
             label_fields=['labels'])
        )

        data_transforms_test = A.Compose([
            A.Normalize( #mean std for potsdam dataset from COWC [Calculate also for spot6]
                mean=[0.3442, 0.3708, 0.3476],
                std=[0.1232, 0.1230, 0.1284]
                )],
            additional_targets={
                 'image_lq':'image'
                })

        self.data_dir_gt = os.path.abspath(data_dir_GT) + '/'
        self.data_dir_lq = os.path.abspath(data_dir_LQ) + '/'

        if training == True:
            self.dataset = data.COWCDataset(self.data_dir_gt, self.data_dir_lq, transform=data_transforms_train)
        else:
            self.dataset = data.COWCDataset(self.data_dir_gt, self.data_dir_lq, transform=data_transforms_test)
        self.length = len(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

def collate_fn(batch):
    '''
    Image have a different number of objects, we need a collate function
    (to be passed to the DataLoader).
    '''
    target = list()
    image = {}
    image['object'] = list()
    image['image'] = list()
    image['image_lq'] = list()
    image['LQ_path'] = list()

    for obj in batch:
        b = obj[0]
        image['object'].append(b['object'])
        image['image'].append(b['image'])
        image['image_lq'].append(b['image_lq'])
        image['LQ_path'].append(b['LQ_path'])
        target.append(obj[1])

    image['object'] = torch.stack(image['object'], dim=0)
    image['image'] = torch.stack(image['image'], dim=0)
    image['image_lq'] = torch.stack(image['image_lq'], dim=0)

    return image, target