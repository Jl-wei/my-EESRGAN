import os
import sys
import cv2
import numpy as np
import shutil
import random


try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import imresize_np
except ImportError:
    pass

class ImagePrepare():
    def __init__(self, source, dest):
        self.up_scale = 4
        self.mod_scale = 4
        self.sourcedir = source
        self.savedir = dest
        self.valid_percent = 0.2

    def generate_lr_hr_bic(self):
        self.mkdir()
        self.generate_img()
        self.copy_annotations()
        self.split_for_valid()

    def mkdir(self):
        self.hr_path = os.path.join(self.savedir, 'HR', 'x' + str(self.mod_scale))
        self.lr_path = os.path.join(self.savedir, 'LR', 'x' + str(self.up_scale))
        self.bic_path = os.path.join(self.savedir, 'Bic', 'x' + str(self.up_scale))

        self.valid_hr_path = os.path.join(self.hr_path, 'valid')
        self.valid_lr_path = os.path.join(self.lr_path, 'valid')
        self.valid_bic_path = os.path.join(self.bic_path, 'valid')

        shutil.rmtree(self.hr_path, ignore_errors=True)
        shutil.rmtree(self.lr_path, ignore_errors=True)
        shutil.rmtree(self.bic_path, ignore_errors=True)

        os.makedirs(self.valid_hr_path, exist_ok=True)
        os.makedirs(self.valid_lr_path, exist_ok=True)
        os.makedirs(self.valid_bic_path, exist_ok=True)

    def generate_img(self):
        filepaths = [f for f in os.listdir(self.sourcedir) if f.endswith('.jpg') and not f.endswith('check.jpg')]
        filepaths.sort()
        num_files = len(filepaths)
        # prepare data with augementation
        for i in range(num_files):
            filename = filepaths[i]
            print('No.{} -- Processing {}'.format(i, filename))
            # read image
            image = cv2.imread(os.path.join(self.sourcedir, filename))

            width = int(np.floor(image.shape[1] / self.mod_scale))
            height = int(np.floor(image.shape[0] / self.mod_scale))
            # modcrop
            if len(image.shape) == 3:
                image_HR = image[0:self.mod_scale * height, 0:self.mod_scale * width, :]
            else:
                image_HR = image[0:self.mod_scale * height, 0:self.mod_scale * width]
            # LR
            image_LR = imresize_np(image_HR, 1 / self.up_scale, True)
            # bic
            image_Bic = imresize_np(image_LR, self.up_scale, True)

            cv2.imwrite(os.path.join(self.hr_path, filename), image_HR)
            cv2.imwrite(os.path.join(self.lr_path, filename), image_LR)
            cv2.imwrite(os.path.join(self.bic_path, filename), image_Bic)

    def copy_annotations(self):
        filepaths = [f for f in os.listdir(self.sourcedir) if f.endswith('.txt')]
        filepaths.sort()
        num_files = len(filepaths)
        # prepare data with augementation
        for i in range(num_files):
            filename = filepaths[i]
            print('No.{} -- Processing {}'.format(i, filename))
            
            origin_file = os.path.join(self.sourcedir, filename)
            shutil.copy2(origin_file, self.hr_path)
            shutil.copy2(origin_file, self.lr_path)
            shutil.copy2(origin_file, self.bic_path)

    def split_for_valid(self):
        random.seed(123)

        file_names = [os.path.splitext(f)[0] for f in os.listdir(self.sourcedir) if f.endswith('.txt')]
        image_nb = len(file_names)
        valid_image_nb = int(image_nb * self.valid_percent)
        valid_file_names = random.sample(file_names, valid_image_nb)

        for filename in valid_file_names:
            shutil.move(os.path.join(self.hr_path, filename) + '.jpg', self.valid_hr_path)
            shutil.move(os.path.join(self.hr_path, filename) + '.txt', self.valid_hr_path)
            shutil.move(os.path.join(self.lr_path, filename) + '.jpg', self.valid_lr_path)
            shutil.move(os.path.join(self.lr_path, filename) + '.txt', self.valid_lr_path)
            shutil.move(os.path.join(self.bic_path, filename) + '.jpg', self.valid_bic_path)
            shutil.move(os.path.join(self.bic_path, filename) + '.txt', self.valid_bic_path)





if __name__ == "__main__":
    # source = 'dataset/Potsdam_ISPRS'
    # dest = 'dataset/Potsdam_ISPRS'

    source = 'dataset/PCB_DATASET/splited'
    dest = 'dataset/PCB_DATASET/splited'
    ImagePrepare(source, dest).generate_lr_hr_bic()

    # import pdb; pdb.set_trace()