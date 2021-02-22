import os
import sys
import cv2
import numpy as np
import shutil

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import imresize_np
except ImportError:
    pass

class PrepareHRIPCB():
    def __init__(self):
        self.up_scale = 8
        self.mod_scale = 8
        self.sourcedir = 'dataset/PCB_DATASET'
        self.savedir = 'dataset/PCB_DATASET/EESRGAN'

    def generate_lr_hr_bic(self):
        self.mkdir()
        self.copy_img()
        self.generate_img()
        self.copy_annotations()

    def mkdir(self):
        os.makedirs(self.savedir, exist_ok=True)

        self.hr_path = os.path.join(self.savedir, 'HR', 'x' + str(self.mod_scale))
        self.lr_path = os.path.join(self.savedir, 'LR', 'x' + str(self.up_scale))
        self.bic_path = os.path.join(self.savedir, 'Bic', 'x' + str(self.up_scale))

        os.makedirs(self.hr_path, exist_ok=True)
        os.makedirs(self.lr_path, exist_ok=True)
        os.makedirs(self.bic_path, exist_ok=True)

    def copy_img(self):
        source_image_path = os.path.join(self.sourcedir, 'images')
        for folder in os.listdir(source_image_path):
            for img in os.listdir(os.path.join(source_image_path, folder)):
                origin_file = os.path.join(source_image_path, folder, img)
                shutil.copy2(origin_file, self.savedir)

    def generate_img(self):
        img_paths = [f for f in os.listdir(self.savedir) if f.endswith('.jpg')]
        # prepare data with augementation
        for i in range(len(img_paths)):
            img_name = img_paths[i]
            print('No.{} -- Processing {}'.format(i, img_name))

            image = cv2.imread(os.path.join(self.savedir, img_name))

            width = int(np.floor(image.shape[1] / self.mod_scale))
            height = int(np.floor(image.shape[0] / self.mod_scale))

            image_HR = image[0:self.mod_scale * height, 0:self.mod_scale * width, :]
            image_LR = imresize_np(image_HR, 1 / self.up_scale, True)
            image_Bic = imresize_np(image_LR, self.up_scale, True)

            cv2.imwrite(os.path.join(self.hr_path, img_name), image_HR)
            cv2.imwrite(os.path.join(self.lr_path, img_name), image_LR)
            cv2.imwrite(os.path.join(self.bic_path, img_name), image_Bic)

    def copy_annotations(self):
        source_annotations_path = os.path.join(self.sourcedir, 'Annotations')
        for folder in os.listdir(source_annotations_path):
            for anno in os.listdir(os.path.join(source_annotations_path, folder)):
                origin_file = os.path.join(source_annotations_path, folder, anno)
                shutil.copy2(origin_file, self.hr_path)
                shutil.copy2(origin_file, self.lr_path)
                shutil.copy2(origin_file, self.bic_path)



if __name__ == "__main__":
    PrepareHRIPCB().generate_lr_hr_bic()