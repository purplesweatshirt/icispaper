from pdf2image import convert_from_path, convert_from_bytes
from google.colab.patches import cv2_imshow
from keras.models import load_model
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import darknet, darknet_images
import numpy as np
import random
import os
import cv2
import sys


def store_images(path_to_pdf):
    """
    input: path to the pdf file
    converts the pages to jpeg images and store them on disk.
    """
    os.system('mkdir temp_imgs')

    pageNumber = 0
    images = convert_from_path(path_to_pdf, fmt = 'jpeg', thread_count = 4)
    for image in images:
        filenames = []
        pageNumber += 1
        filename = f"./temp_imgs/Page{pageNumber}.jpg"
        image.save(filename, 'JPEG')
    del images


def classify_pages(model_path='backup/sem_class.h5', img_path='./temp_imgs/', thresh=0.5):
    """
    input: path to model h5-file, path to image folder, threshold for classification
    classifies (sem or no sem) each page of the pdf and deletes the no sem pages
    """

    model = load_model(model_path)

    for file in os.scandir(img_path):
        if file.is_file() and file.name.endswith('.jpg'):
            print(img_path+file.name)
            img = Image.open(img_path+file.name,'r')
            img = img.resize((512, 512), Image.ANTIALIAS)
            img_arr = np.array(img)
            img_arr = img_arr/255
            img.close()
            img_arr = np.reshape(img_arr, (1, img_arr.shape[0], img_arr.shape[1], 3))
            y_prob = model.predict(img_arr)
            if y_prob > thresh:
                os.remove(img_path+file.name)             


def detect_figures(cfg_file='cfg/fig_det.cfg',
                   data='data/fig_det.data',
                   weights='backup/fig_det.weights',
                   path_to_imgs='./temp_imgs',
                   dest_path='./cropped_imgs'):
    """
    input: path to cfg-file, path to data-file, path to weights, path to image folder
    applies yolov4 model to each image to detect SEM figures in a given pdf page
    """
    os.system(f'mkdir {dest_path}')

    for id in os.listdir(path_to_imgs):
        if id[-3:] == 'jpg':
            print(f"Processing image {id[:-4]}")
            path = path_to_imgs + '/' + id
            path_to_txt = path_to_imgs + '/' + id[:-3] + 'txt'
            os.system(f'./darknet detector test {data} {cfg_file} {weights} {path} -save_labels -dont_show')

            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            width, height = Image.fromarray(image, 'RGB').convert('L').size[0], Image.fromarray(image, 'RGB').convert('L').size[1]
                        
            with open(path_to_txt, 'r') as f:
                contents = f.read()
                contents = [line.split(' ') for line in contents.split('\n') if len(line) > 0]
                detections = [[width*float(line[1]), height*float(line[2]), width*float(line[3]), height*float(line[4])] for line in contents]
                detections = [[contents[i][0], darknet.bbox2points(detections[i])] for i in range(len(detections)) if contents[i][0]=='0']
                for i in range(len(detections)):
                    x1, y1, x2, y2 = detections[i][1]
                    roi = image[y1:y2, x1:x2]
                    cv2.imwrite(dest_path+f'/{i}_'+id, roi)
                    cv2_imshow(roi)


def correct_confusions(detections):
    """
    input: list of all detections
    determine whether there are no constructs detected in the image. If so, swap labels
    of item and construct prediction
    """
    const = any([detections[i][0] == 'c' for i in range(len(detections))])
    if not const:
        for i in range(len(detections)):
            if detections[i][0] == 'p':
                detections[i][0] = 'c'
    
    return detections



def detect_variables(cfg_file='cfg/var_det.cfg',
                     data='data/var_det.data',
                     weights='backup/variable_detection.weights',
                     path_to_imgs='./cropped_imgs',
                     dest_path='./final_imgs'):
    """
    input: path to cfg-file, path to data-file, path to weights, path to image folder
    applies yolov4 model to each image to detect variables and path coefficients
    in a (cropped) SEM figure
    """
    os.system(f'mkdir {dest_path}')
    
    colors = {'c': (249, 69, 252), 'i': (241, 200, 98), 'p': (88, 255, 145)}

    for id in os.listdir(path_to_imgs):
        if id[-3:] == 'jpg':
            print(f"Processing image {id[:-4]}")
            path = path_to_imgs + '/' + id
            path_to_txt = path_to_imgs + '/' + id[:-3] + 'txt'
            os.system(f'./darknet detector test {data} {cfg_file} {weights} {path} -save_labels -dont_show')

            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            width, height = Image.fromarray(image, 'RGB').convert('L').size[0], Image.fromarray(image, 'RGB').convert('L').size[1]

            with open(path_to_txt, 'r') as f:
                contents = f.read()
                contents = [line.split(' ') for line in contents.split('\n') if len(line) > 0]
                detections = [[width*float(line[1]), height*float(line[2]), width*float(line[3]), height*float(line[4])] for line in contents]
                detections = [[contents[i][0], darknet.bbox2points(detections[i])] for i in range(len(detections))]
                detections = correct_confusions(detections)
                for i in range(len(detections)):
                    x1, y1, x2, y2 = detections[i][1]
                    if detections[i][0] == '0':
                      cv2.rectangle(image, (x1, y1), (x2, y2), color=colors['c'], thickness=2)
                    elif detections[i][0] == '1':
                      cv2.rectangle(image, (x1, y1), (x2, y2), color=colors['i'], thickness=2)
                    else:
                      cv2.rectangle(image, (x1, y1), (x2, y2), color=colors['p'], thickness=2)
                cv2.imwrite(dest_path+'/'+id, image)
                cv2_imshow(image)
