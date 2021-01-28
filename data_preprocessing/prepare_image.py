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
        self.sourcedir = '/Users/jlwei/workspace/datasets/DetectionPatches_256x256/Potsdam_ISPRS'
        self.savedir = '/Users/jlwei/workspace/datasets/DetectionPatches_256x256/Potsdam_ISPRS'

    def generate_lr_hr_bic(self):
        self.mkdir()
        self.generate_img()
        self.copy_annotations()

    def mkdir(self):
        self.saveHRpath = os.path.join(self.savedir, 'HR', 'x' + str(self.mod_scale))
        self.saveLRpath = os.path.join(self.savedir, 'LR', 'x' + str(self.up_scale))
        self.saveBicpath = os.path.join(self.savedir, 'Bic', 'x' + str(self.up_scale))

        if not os.path.isdir(self.sourcedir):
            print('Error: No source data found')
            exit(0)
        if not os.path.isdir(self.savedir):
            os.mkdir(self.savedir)

        if not os.path.isdir(os.path.join(self.savedir, 'HR')):
            os.mkdir(os.path.join(self.savedir, 'HR'))
        if not os.path.isdir(os.path.join(self.savedir, 'LR')):
            os.mkdir(os.path.join(self.savedir, 'LR'))
        if not os.path.isdir(os.path.join(self.savedir, 'Bic')):
            os.mkdir(os.path.join(self.savedir, 'Bic'))

        if not os.path.isdir(self.saveHRpath):
            os.mkdir(self.saveHRpath)
        else:
            print('It will cover ' + str(self.saveHRpath))

        if not os.path.isdir(self.saveLRpath):
            os.mkdir(self.saveLRpath)
        else:
            print('It will cover ' + str(self.saveLRpath))

        if not os.path.isdir(self.saveBicpath):
            os.mkdir(self.saveBicpath)
        else:
            print('It will cover ' + str(self.saveBicpath))

    def generate_img(self):
        filepaths = [f for f in os.listdir(self.sourcedir) if f.endswith('.jpg') and not f.endswith('check.jpg')]
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

            cv2.imwrite(os.path.join(self.saveHRpath, filename), image_HR)
            cv2.imwrite(os.path.join(self.saveLRpath, filename), image_LR)
            cv2.imwrite(os.path.join(self.saveBicpath, filename), image_Bic)

    def copy_annotations(self):
        filepaths = [f for f in os.listdir(self.sourcedir) if f.endswith('.txt')]
        num_files = len(filepaths)
        # prepare data with augementation
        for i in range(num_files):
            filename = filepaths[i]
            print('No.{} -- Processing {}'.format(i, filename))
            
            origin_file = os.path.join(self.sourcedir, filename)
            shutil.copy2(origin_file, self.saveHRpath)
            shutil.copy2(origin_file, self.saveLRpath)
            shutil.copy2(origin_file, self.saveBicpath)




if __name__ == "__main__":
    PrepareImage().generate_lr_hr_bic()