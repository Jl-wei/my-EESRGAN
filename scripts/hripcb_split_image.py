import cv2
import numpy as np
import sys
import os
import shutil
import xml.etree.ElementTree as ET

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import imresize_np
except ImportError:
    pass

def start_points(size, split_size, overlap):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def split_img(img_path, annot_path, dest_path):
    labels = {'missing_hole': 1, 'mouse_bite': 2, 'open_circuit': 3,
            'short': 4, 'spur': 5, 'spurious_copper': 6}

    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    split_width = 256
    split_height = 256

    X_points = start_points(img_w, split_width, 0.4)
    Y_points = start_points(img_h, split_height, 0.4)

    root = ET.parse(annot_path).getroot()

    filename = root.find('filename').text.split(".")[0]
    objects = root.findall('object')

    boxes = list()

    # Get annotation
    for obj in objects:
        defects_type = int(labels[obj.find('name').text])

        bnbbox = obj.find('bndbox')
        x_min = int(bnbbox.find('xmin').text)
        y_min = int(bnbbox.find('ymin').text)
        x_max = int(bnbbox.find('xmax').text)
        y_max = int(bnbbox.find('ymax').text)
        boxes.append([y_min, y_max, x_min, x_max, defects_type])

    # Split image
    count = 1
    for i in Y_points:
        for j in X_points:
            tile_boxes = []
            for box in boxes:
                # boxes in this tile
                if i <= box[0] and box[1] <= i+split_height and \
                    j <= box[2] and box[3] <= j+split_width:
                    tile_boxes.append(box)

            if tile_boxes:
                annot_name = os.path.join(dest_path , '{}_{}.txt'.format(filename, count))

                with open(annot_name, 'w') as f:
                    for box in tile_boxes:
                        defects_type = box[4]
                        box_w_p = (box[3] - box[2]) / split_width
                        box_h_p = (box[1] - box[0]) / split_height
                        x_cen_p = (box[2] - j) / split_width + 0.5 * box_w_p
                        y_cen_p = (box[0] - i) / split_height + 0.5 * box_h_p

                        f.write("{} {} {} {} {}\n".format(defects_type, x_cen_p, y_cen_p, box_w_p, box_h_p))

                image_name = os.path.join(dest_path , '{}_{}.jpg'.format(filename, count))
                split = img[i:i+split_height, j:j+split_width]
                # split = imresize_np(split, 0.8, True)
                cv2.imwrite(image_name, split)    
                count += 1

def copy_imgs(dataset_dir, origin_dir):
    source_image_path = os.path.join(dataset_dir, 'images')
    for folder in os.listdir(source_image_path):
        for img in os.listdir(os.path.join(source_image_path, folder)):
            origin_file = os.path.join(source_image_path, folder, img)
            shutil.copy2(origin_file, origin_dir)

def copy_annotations(dataset_dir, origin_dir):
    source_annotations_path = os.path.join(dataset_dir, 'Annotations')
    for folder in os.listdir(source_annotations_path):
        for anno in os.listdir(os.path.join(source_annotations_path, folder)):
            origin_file = os.path.join(source_annotations_path, folder, anno)
            shutil.copy2(origin_file, origin_dir)

# It splits images into 256x256 tiles for the input of GAN
def split_imgs(origin_dir, split_dir):
    filepaths = [f for f in os.listdir(origin_dir) if f.endswith('.jpg')]

    for i in range(len(filepaths)):
        filename = os.path.splitext(filepaths[i])[0]
        print('No.{} -- Processing {}'.format(i, filename))
        img_path = os.path.join(origin_dir, filename) + '.jpg'
        annot_path = os.path.join(origin_dir, filename) + '.xml'
        split_img(img_path, annot_path, split_dir)

def count_labels_xml(dir):
    labels_count = {'missing_hole': 0, 'mouse_bite': 0, 'open_circuit': 0,
                    'short': 0, 'spur': 0, 'spurious_copper': 0}
    for f in os.listdir(dir):
        if f.endswith('.xml'):
            root = ET.parse(os.path.join(dir, f)).getroot()
            objects = root.findall('object')

            # Get annotation
            for obj in objects:
                defects_type = obj.find('name').text
                labels_count[defects_type] += 1

    print(labels_count)

def count_labels_txt(dir):
    labels_count = {'missing_hole': 0, 'mouse_bite': 0, 'open_circuit': 0,
                    'short': 0, 'spur': 0, 'spurious_copper': 0}
    for f in os.listdir(dir):
        if f.endswith('.txt'):
            with open(os.path.join(dir, f)) as txt:
                for line in txt:
                    values = (line.split())
                    if values[0] == '1':
                        labels_count['missing_hole'] += 1
                    elif values[0] == '2':
                        labels_count['mouse_bite'] += 1
                    elif values[0] == '3':
                        labels_count['open_circuit'] += 1
                    elif values[0] == '4':
                        labels_count['short'] += 1
                    elif values[0] == '5':
                        labels_count['spur'] += 1
                    elif values[0] == '6':
                        labels_count['spurious_copper'] += 1

    print(labels_count)


if __name__ == "__main__":
    dataset_dir = 'dataset/PCB_DATASET'
    origin_dir = 'dataset/PCB_DATASET/original/'
    split_dir = 'dataset/PCB_DATASET/splited/'

    shutil.rmtree(origin_dir, ignore_errors=True)
    os.mkdir(origin_dir)
    shutil.rmtree(split_dir, ignore_errors=True)
    os.mkdir(split_dir)

    copy_imgs(dataset_dir, origin_dir)
    copy_annotations(dataset_dir, origin_dir)
    split_imgs(origin_dir, split_dir)

    print('original')
    count_labels_xml(origin_dir)
    print('splited')
    count_labels_txt(split_dir)

    if os.path.isdir(origin_dir + 'HR/x4/'):
        print('train')
        count_labels_xml(origin_dir + 'HR/x4/')
        print('valid')
        count_labels_txt(split_dir + 'HR/x4/valid/')