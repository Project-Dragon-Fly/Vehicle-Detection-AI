import cv2
import pandas as pd

CONST = dict()
CONST['camera'] = "camera/data/camera.csv"
CONST['link']   = "camera/data/cam_link.csv"


# ------------------------------------------------------------------------------
class VideoExtractor:
    def __init__(self, video_in_path, video_start_time, start_time, end_time):
        self.video = cv2.VideoCapture(video_in_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.cur_time = video_start_time
        self.end_time = end_time

        ret, frame = self.video.read()
        while ret and (self.cur_time < start_time):
            self.cur_time += 1/self.fps
            ret, frame = self.video.read()

        if not ret:
            self.__del__()

    def __del__(self):
        self.video.release()

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.video.read()
        if ret and (self.cur_time <= self.end_time):
            self.cur_time += 1/self.fps
            return frame
        self.__del__()
        raise StopIteration

class Camera:
    def __init__(self, id, loc, local_name, division, video_loc, video_start_time):
        self.id = id
        self.location = loc
        self.local_name = local_name
        self.division_name = division
        self.video_loc = video_loc
        self.video_start_time = video_start_time
        video = cv2.VideoCapture(self.video_loc);
        self.fps = video.get(cv2.CAP_PROP_FPS)
        video.release()


        self.link = dict()
        self.update_link(self,0,216000) # self reference, min=0sec, max=1hr

    def __str__(self):
        return f"camera {self.id} at {self.location} {self.local_name} {self.division_name}"

    def update_link(self,camera,min_time, max_time):
        lnk = dict()
        lnk['camera'] = camera
        lnk['min_time'] = min_time
        lnk['max_time'] = max_time
        self.link[camera.id] = lnk

    def next_camera(self):
        return self.link

    def filter_frame(self,start_time,end_time):
        return VideoExtractor(self.video_loc,self.video_start_time, start_time, end_time)

#-------------------------------------------------------------------------------

# LOADING DATA INTO CALSS OBJ---------------------------------------------------
camera_obj = dict()

DB = pd.read_csv(CONST['camera'])
for j in range(len(DB)):
    id         = DB['camera'][j]
    loc        = DB['location cord'][j]
    local_name = DB['local name'][j]
    division   = DB['division name'][j]
    video_loc  = DB['video_loc'][j]
    v_start_time = DB["video_start_time"][j]

    camera_obj[id] = Camera(id,loc,local_name,division,video_loc,v_start_time)

#------------------------------------------------------------

DB = pd.read_csv(CONST['link'])
for j in range(len(DB)):
    src   = DB['from_cam_ip'][j]
    des   = DB['to_cam_ip'][j]
    min_t = DB['min_time'][j]
    max_t = DB['max_time'][j]

    cam1 = camera_obj[src]
    cam2 = camera_obj[des]

    cam1.update_link(cam2,min_t,max_t)
    cam2.update_link(cam1,min_t,max_t)
#-------------------------------------------------------------------------------
