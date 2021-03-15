import cv2
import sys

path = sys.argv[1]

img_path = path + ".jpg"
anno_path = path + ".txt"

img = cv2.imread(img_path)
with open(anno_path) as f:
    for line in f:
        values = (line.split())
        #get coordinates withing height width range
        x = float(values[1]) * 256
        y = float(values[2]) * 256
        width = float(values[3]) * 256
        height = float(values[4]) * 256

        #creating bounding boxes that would not touch the image edges
        x_min = 1 if x - width/2 <= 0 else int(x - width/2)
        x_max = 255 if x + width/2 >= 256 else int(x + width/2)
        y_min = 1 if y - height/2 <= 0 else int(y - height/2)
        y_max = 255 if y + height/2 >= 256 else int(y + height/2)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0), 2)

cv2.imwrite("./test_img.jpg", img)


'''
python scripts/draw_box.py dataset/PCB_DATASET/splited/01_spur_04_4
'''