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

class PrepareImage():
    def __init__(self):
        self.up_scale = 4
        self.mod_scale = 4
        self.sourcedir = 'dataset/Potsdam_ISPRS'
        self.savedir = 'dataset/Potsdam_ISPRS'
        self.valid_percent = 0.2

    def generate_lr_hr_bic(self):
        self.mkdir()
        self.generate_img()
        self.copy_annotations()

    def mkdir(self):
        self.hr_path = os.path.join(self.savedir, 'HR', 'x' + str(self.mod_scale))
        self.lr_path = os.path.join(self.savedir, 'LR', 'x' + str(self.up_scale))
        self.bic_path = os.path.join(self.savedir, 'Bic', 'x' + str(self.up_scale))

        self.valid_hr_path = os.path.join(self.hr_path, 'valid')
        self.valid_lr_path = os.path.join(self.lr_path, 'valid')
        self.valid_bic_path = os.path.join(self.bic_path, 'valid')

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

            if i < num_files * (1 - self.valid_percent):
                cv2.imwrite(os.path.join(self.hr_path, filename), image_HR)
                cv2.imwrite(os.path.join(self.lr_path, filename), image_LR)
                cv2.imwrite(os.path.join(self.bic_path, filename), image_Bic)
            else:
                cv2.imwrite(os.path.join(self.valid_hr_path, filename), image_HR)
                cv2.imwrite(os.path.join(self.valid_lr_path, filename), image_LR)
                cv2.imwrite(os.path.join(self.valid_bic_path, filename), image_Bic)

    def copy_annotations(self):
        filepaths = [f for f in os.listdir(self.sourcedir) if f.endswith('.txt')]
        filepaths.sort()
        num_files = len(filepaths)
        # prepare data with augementation
        for i in range(num_files):
            filename = filepaths[i]
            print('No.{} -- Processing {}'.format(i, filename))
            
            origin_file = os.path.join(self.sourcedir, filename)
            if i < num_files * (1 - self.valid_percent):
                shutil.copy2(origin_file, self.hr_path)
                shutil.copy2(origin_file, self.lr_path)
                shutil.copy2(origin_file, self.bic_path)
            else:
                shutil.copy2(origin_file, self.valid_hr_path)
                shutil.copy2(origin_file, self.valid_lr_path)
                shutil.copy2(origin_file, self.valid_bic_path)




if __name__ == "__main__":
    PrepareImage().generate_lr_hr_bic()