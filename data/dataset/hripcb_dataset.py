from __future__ import print_function, division
import os
import torch
import numpy as np
import glob
import cv2
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class HRIPCBDataset(Dataset):
  def __init__(self, data_dir_gt, data_dir_lq, image_height=256, image_width=256, transform = None):
    self.data_dir_gt = data_dir_gt
    self.data_dir_lq = data_dir_lq
    #take all under same folder for train and test split.
    self.transform = transform
    self.image_height = image_height
    self.image_width = image_width
    #sort all images for indexing, filter out check.jpgs
    self.imgs_gt = list(sorted(glob.glob(self.data_dir_gt+"*.jpg")))
    self.imgs_lq = list(sorted(glob.glob(self.data_dir_lq+"*.jpg")))
    self.annotation = list(sorted(glob.glob(self.data_dir_lq+"*.txt")))

    self.labels = {'missing_hole': 1, 'mouse_bite': 2, 'open_circuit': 3,
                    'short': 4, 'spur': 5, 'spurious_copper': 6}

  def __getitem__(self, idx):
    #get the paths
    img_path_gt = os.path.join(self.data_dir_gt, self.imgs_gt[idx])
    img_path_lq = os.path.join(self.data_dir_lq, self.imgs_lq[idx])
    annotation_path = os.path.join(self.data_dir_lq, self.annotation[idx])
    img_gt = cv2.imread(img_path_gt,1) #read color image height*width*channel=3
    img_lq = cv2.imread(img_path_lq,1) #read color image height*width*channel=3
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

    root = ET.parse(annotation_path).getroot()

    filename = root.find('filename').text
    objects = root.findall('object')

    boxes = list()
    defects_type = list()

    for obj in objects:
        defects_type.append(self.labels[obj.find('name').text])

        bnbbox = obj.find('bndbox')
        x_min = bnbbox.find('xmin').text
        y_min = bnbbox.find('ymin').text
        x_max = bnbbox.find('xmax').text
        y_max = bnbbox.find('ymax').text
        boxes.append([x_min, y_min, x_max, y_max])


    if obj_class != 0:
        labels = np.ones(len(boxes)) # all are cars
        boxes_for_calc = torch.as_tensor(boxes, dtype=torch.int64)
        area = (boxes_for_calc[:, 3] - boxes_for_calc[:, 1]) * (boxes_for_calc[:, 2] - boxes_for_calc[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        #create dictionary to access the values
        target = {}
        target['object'] = 1
        target['image_lq'] = img_lq
        target['image'] = img_gt
        target['bboxes'] = boxes
        target['labels'] = labels
        target['label_car_type'] = label_car_type
        target['image_id'] = idx
        target['LQ_path'] = img_path_lq
        target["area"] = area
        target["iscrowd"] = iscrowd

    if self.transform is None:
        #convert to tensor
        image, target = self.convert_to_tensor(**target)
        return image, target
        #transform
    else:
        transformed = self.transform(**target)
        #print(transformed['image'], transformed['bboxes'], transformed['labels'], transformed['idx'])
        image, target = self.convert_to_tensor(**transformed)
        return image, target

  def __len__(self):
    return len(self.imgs_lq)

  def convert_to_tensor(self, **target):
      #convert to tensor
      target['object'] = torch.tensor(target['object'], dtype=torch.int64)
      target['image_lq'] = torch.from_numpy(target['image_lq'].transpose((2, 0, 1)))
      target['image'] = torch.from_numpy(target['image'].transpose((2, 0, 1)))
      target['boxes'] = torch.tensor(target['bboxes'], dtype=torch.float32)
      target['labels'] = torch.ones(len(target['labels']), dtype=torch.int64)
      target['label_car_type'] = torch.tensor(target['label_car_type'], dtype=torch.int64)
      target['image_id'] = torch.tensor([target['image_id']])
      target["area"] = torch.tensor(target['area'])
      target["iscrowd"] = torch.tensor(target['iscrowd'])

      image = {}
      image['object'] = target['object']
      image['image_lq'] = target['image_lq']
      image['image'] = target['image']
      image['image'] = target['image']
      image['LQ_path'] = target['LQ_path']

      del target['object']
      del target['image_lq']
      del target['image']
      del target['bboxes']
      del target['LQ_path']

      return image, target
