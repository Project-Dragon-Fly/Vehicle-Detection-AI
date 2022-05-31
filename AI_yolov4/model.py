import logging
import cv2
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4

CONSTS = dict()
CONSTS['cfg'] = 'AI_yolov4/data/pip-yolov4-vehicle.cfg'
CONSTS['names'] = 'AI_yolov4/data/classes.names'
CONSTS['weights'] = 'AI_yolov4/data/yolov4-vehicle_best.weights'


yolo = YOLOv4()
yolo.config.parse_names(CONSTS['names'])
yolo.config.parse_cfg(CONSTS['cfg'])


def make_model():
    global yolo
    try:
        yolo.model
    except AttributeError:
        logging.info("loading the model weights")
        yolo.make_model()
        yolo.load_weights(CONSTS['weights'], weights_type="yolo")
        

def predict(frame,min_threshold=0.5):
    make_model()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pred_bbox = yolo.predict(frame,min_threshold)
    pred_bbox = pred_bbox[pred_bbox[:, 5] > min_threshold]
    bboxes = pred_bbox[:,0:4] #  x,y,w,h
    names = np.array([yolo.config.names[int(i)] for i in pred_bbox[:,4]])
    scores = pred_bbox[:,5]
    return (names,bboxes,scores)

def get_valid_class_label():
    return [ yolo.config.names[i] for i in yolo.config.names]
