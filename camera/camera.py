import logging
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

CONSTS = dict()
CONSTS['camera'] = "camera/data/camera.csv"

class Camera:
    def __init__(self, cam_id, cam_name, latitude, longitude, detection_file, start_time):
        self.cam_id = int(cam_id)
        self.cam_name = str(cam_name)
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.start_time = float(start_time)
        self.video_file = str(video_file)
        self.detection_file = str(detection_file)
        self.saved_frame = str(saved_frame)
        
        self.distance = dict()

    def __str__(self):
        return f"Camera {self.cam_name} at ({self.latitude},{self.longitude})"

    def save_gps_distance(self, camera, threshold_KM=2):
        lon1 = radians(self.longitude)
        lon2 = radians(camera.longitude)
        lat1 = radians(self.latitude)
        lat2 = radians(camera.latitude)
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        distance = c*r # KM
        
        if distance <= threshold_KM:
            self.distance[camera.cam_id] = distance
    

    def next_camera(self):
        id_list = sorted(self.distance, key=lambda k: self.distance[k])
        parse_cam = dict()
        for cid in id_list:
            parse_cam[cid] = dict()
            parse_cam[cid]['distance'] = self.distance[cid] # in KM
            parse_cam[cid]['min_time'] = (self.distance[cid]/100)*60*60  # max speed 100 kmph
            parse_cam[cid]['max_time'] = (self.distance[cid]/20)*60*60  # min speed 20 kmph
        return parse_cam

    def filter_frame(self,start_time,end_time,class_label):
        detections = pd.read_csv(self.detection_file)
        
        prune = detections[detections['time_sec'] >= start_time-self.start_time]
        prune = prune[prune['time_sec'] <= end_time-self.start_time]     
        prune = prune[prune['class_label']==class_label]
        
        unique_id = prune['id'].unique()
        vehicles = list()
        for uid in unique_id:
            vehi = dict()
            vehi['cam_id'] = self.cam_id
            vehi['class_label'] = class_label
            prune = detections[detections['id']==uid]
            vehi['entry_time'] = prune['time_sec'].min() + self.start_time
            vehi['exit_time'] = prune['time_sec'].max() + self.start_time
            vehi['bbox'] = prune['bbox'].iloc[0]
            vehi['bbox'] = list(map(float,vehi['bbox'][1:-1].split()))
            
            vehicles.append(vehi)
        
        return vehicles
            

            
#-------------------------------------------------------------------------------
# LOADING DATA INTO CALSS OBJ---------------------------------------------------
logging.info(f"creating camera objects... loading {CONSTS['camera']}")
camera_obj = dict()

DB = pd.read_csv(CONSTS['camera'])
for j in range(len(DB)):
    cam_id         = DB['camera_id'][j]
    cam_name       = DB['camera_name'][j]
    latitude       = DB['latitude'][j]
    longitude      = DB['longitude'][j]
    start_time     = DB['start_time'][j]
    video_file     = DB['video_file'][j]
    detection_file = DB['detection_file'][j]
    saved_frame    = DB['saved_frame'][j]
    
    

    C = Camera(cam_id,cam_name,latitude,longitude,start_time,video_file,detection_file,saved_frame)
    for cid in camera_obj:
        C.save_gps_distance(camera_obj[cid])
        camera_obj[cid].save_gps_distance(C)
        
    camera_obj[cam_id] = C