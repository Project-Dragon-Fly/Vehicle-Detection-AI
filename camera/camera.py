import cv2
import pandas as pd

CONST = dict()
CONST['camera'] = "data/camera.csv"
CONST['link']   = "data/cam_link.csv"


# ------------------------------------------------------------------------------

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

    def update_link(self,camera,min_time, max_time):
        lnk = dict()
        lnk['camera'] = camera
        lnk['min_time'] = min_time
        lnk['max_time'] = max_time
        self.link[camera.id] = lnk

    def next_camera(self):
        return self.link

    def filter_frame(self,start_time,end_time):
        frames = list()
        video = cv2.VideoCapture(self.video_loc);
        cur_time = self.video_start_time
        ret, frame = video.read()

        while ret and (cur_time > start_time):
            cur_time += 1/self.fps
            ret, frame = video.read()

        while ret and (cur_time < end_time):
            frames.append(frame)
            ret, frame = video.read()
            cur_time += 1/self.fps

        video.release()
        return frames

#-------------------------------------------------------------------------------

# LOADING DATA INTO CALSS OBJ---------------------------------------------------
camera_obj = dict()

DB = pd.read_csv(CONST['camera'])
for j in range(len(DB)):
    id = DB['camera'][j]
    loc = CAMERA[id]['location cord'] = DB['location cord'][j]
    local_name = CAMERA[id]['local name'] = DB['local name'][j]
    division = CAMERA[id]['division name'] = DB['division name'][j]

    camera_obj[id] = Camera(id,loc,local_name,division)

#------------------------------------------------------------

DB = pd.read_csv(CONST['link'])
for j in range(DB):
    src   = DB['from_cam_ip'][j]
    des   = DB['to_cam_ip'][j]
    min_t = DB['min_time'][j]
    max_t = DB['max_time'][j]

    cam1 = camera_obj[src]
    cam2 = camera_obj[des]

    cam1.update_link(cam2,min_t,max_t)
    cam2.update_link(cam1,min_t,max_t)
#-------------------------------------------------------------------------------
