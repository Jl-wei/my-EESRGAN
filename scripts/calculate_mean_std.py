import cv2
import sys
import os
import shutil
import numpy as np

total_mean = np.array([0.0, 0.0, 0.0])
total_std = np.array([0.0, 0.0, 0.0])

source_dir = "dataset/PCB_DATASET/splited"
filepaths = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
num_files = len(filepaths)

for i in range(num_files):
    filename = filepaths[i]
    file_path = os.path.join(source_dir, filename)

    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean, std = cv2.meanStdDev(img)

    total_mean += np.reshape(np.array(mean), -1)
    total_std += np.reshape(np.array(std), -1)

total_mean = total_mean / num_files
total_std = total_std / num_files

total_mean_p = total_mean / 255
total_std_p = total_std / 255

# import pdb; pdb.set_trace()

print(total_mean)
print(total_std)
print("[{:0.4f}, {:0.4f}, {:0.4f}]".format(total_mean_p[0], total_mean_p[1], total_mean_p[2]))
print("[{:0.4f}, {:0.4f}, {:0.4f}]".format(total_std_p[0], total_std_p[1], total_std_p[2]))
