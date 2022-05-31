"""
Script that converts the video frame into a csv format of detections
"""
from pathlib import Path
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("video_file")
parser.add_argument("csv_file")
parser.add_argument("frame_save_dir")
args = parser.parse_args()

video_file = args.video_file
csv_file = args.csv_file
frame_dir = args.frame_save_dir

import cv2
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from AI_yolov4.model import predict as yolo_predict
from AI_yolov4.deep_sort import preprocessing, nn_matching
from AI_yolov4.deep_sort.detection import Detection
from AI_yolov4.deep_sort.tracker import Tracker
from AI_yolov4.tools import generate_detections as gdet


CONSTS = dict()
CONSTS['deep_sort_model'] =  'AI_yolov4/data/mars-small128.pb'
CONSTS['max_cosine_distance'] = 0.4
CONSTS['nn_budget'] = None
CONSTS['nms_max_overlap'] = 1.0
encoder = gdet.create_box_encoder(CONSTS['deep_sort_model'], batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", CONSTS['max_cosine_distance'], CONSTS['nn_budget'])


logging.info(f"loading the video file {video_file}")
video = cv2.VideoCapture(video_file)
fps = video.get(cv2.CAP_PROP_FPS)
max_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
logging.info(f"loading video file at {fps} fps with lenght {max_frames} frames")


tracker = Tracker(metric)
v_detections = {'id':list(), 'frame_no':list(), 'time_sec':list(), 
                'class_label':list(), 'bbox':list(), 'features':list()
               }

logging.info("extracting the frames")
frame_save_timer = dict()
frame_no = -1
try:
    while True:
        status,frame = video.read()
        if not status:
            logging.info("video file ended")
            break
        frame_no += 1

        names, bboxes, scores = yolo_predict(frame)
        frame_size = frame.shape[:2]

        # change bboxes to x_min,y_min,width,height
        bboxes[:,0] -= bboxes[:,2]/2  # x_min
        bboxes[:,1] -= bboxes[:,3]/2  # y_min
        bboxes[:,0] *= frame_size[1]
        bboxes[:,2] *= frame_size[1]
        bboxes[:,1] *= frame_size[0]
        bboxes[:,3] *= frame_size[0]

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, CONSTS['nms_max_overlap'], scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            v_detections['id'].append(track.track_id)
            v_detections['frame_no'].append(frame_no)
            v_detections['time_sec'].append(frame_no/fps)
            v_detections['class_label'].append(track.class_name)
            v_detections['bbox'].append(track.to_tlbr())
            v_detections['features'].append(track.features)
            
            # save the image
            dir_path = os.path.join(frame_dir,str(track.track_id))
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            _,_,w,h = track.to_tlwh()
            if track.track_id not in frame_save_timer:
                frame_save_timer[track.track_id] = 0
            
            if frame_save_timer[track.track_id] <= 0 and w*h >= 10000:
                file_no = str(len(os.listdir(dir_path))+1)
                file_path = os.path.join(dir_path,f"{file_no}.png")
                ymin,xmin,ymax,xmax = tuple(map(int,track.to_tlbr()))
                cropped_img = frame[xmin:xmax,ymin:ymax]
                try:
                    cv2.imwrite(file_path,cropped_img)
                    frame_save_timer[track.track_id] = 11
                except Exception as e:
                    logging.warning(e)
            
            frame_save_timer[track.track_id] -= 1 
                
            

            logging.info(f"found {track.class_name} at frame no {frame_no}")
        logging.info(f"%5.2f percent {frame_no}/{max_frames}"%(frame_no/max_frames))   
finally:
    logging.info(f"saveing into {csv_file}")
    df = pd.DataFrame(v_detections)
    df.to_csv(csv_file)