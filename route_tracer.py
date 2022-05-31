from camera.camera import camera_obj
from AI_yolov4.config import get_valid_class_label
import pandas as pd
from datetime import datetime


VEHICLE_LABEL = get_valid_class_label()
INITIAL_LOCATION = { i:camera_obj[i].cam_name for i in camera_obj }


class UserAbortedSearch(Exception):
    """ User aborted Request    """
    pass

class VehicleNotFound(Exception):
    """ User aborted Request    """
    pass

class Tracer:
    v_options = VEHICLE_LABEL
    c_options = INITIAL_LOCATION

    def __init__(self, v_type, c_id, start_time, end_time):
        self.v_type = v_type
        self.camera = camera_obj[c_id]
        self.start_time = start_time
        self.end_time = end_time
        self._initialise_trace()
        

    def _initialise_trace(self):
        self.vehicle_history = dict()
        print("scanning initial camera", self.camera)
        self.vehicle_list = self.camera.filter_frame(self.start_time, self.end_time, self.v_type)
        if len(self.vehicle_list) == 0:
            print("Terminating the search")
            self.vehicle = None
            raise VehicleNotFound
        
        elif len(self.vehicle_list) == 1:
            self.vehicle = self.vehicle_list[0]

        elif len(self.vehicle_list) > 1:
            self.vehicle = None
            return self.vehicle_list

    def _append_vehicle_history(self, vehicle):
        for key in vehicle:
            if self.vehicle_history.get(key) is None:
                self.vehicle_history[key] = list()
            else:
                self.vehicle_history[key].append(vehicle[key])


    def trace(self):
        while True: 
            self._append_vehicle_history(self.vehicle)

            cam_id = self.vehicle['cam_id']
            cam = camera_obj[cam_id]
            next_cam_dict = cam.next_camera()
            
            self.vehicle_list = list()
            for nxt_cam_id in next_cam_dict:
                nxt_cam = camera_obj[nxt_cam_id]
                min_time = next_cam_dict[nxt_cam_id]['min_time'] + self.vehicle['exit_time']
                max_time = next_cam_dict[nxt_cam_id]['max_time'] + self.vehicle['exit_time']
                
                print(f"scanning camera {nxt_cam} from {min_time} to {max_time} seconds")
                self.vehicle_list.extend(nxt_cam.filter_frame(min_time, max_time, self.v_type))
            
            if len(self.vehicle_list) == 0:
                print("Terminating the search")
                self.vehicle = None
                raise VehicleNotFound
            
            elif len(self.vehicle_list) == 1:
                self.vehicle = self.vehicle_list[0]

            elif len(self.vehicle_list) > 1:
                self.vehicle = None
                return self.vehicle_list


    def select(self, vehicle_index):
        self.vehicle = self.vehicle_list[vehicle_index]
        self._append_vehicle_history(self.vehicle)
    
    def get_trace_status(self):
        df = pd.DataFrame(self.vehicle_history)
        df = df.sort_values('exit_time')

        print("Travelled route")
        route_data = []
        for index,row in df.iterrows():
            cam = camera_obj[row['cam_id']]
            print(f"{cam} from {row['entry_time']} to {} at {cam.latitude},{cam.longitude}")
            date_str = datetime.fromtimestamp(row['exit_time']).strftime("%m/%d/%Y, %H:%M:%S")
            route_data.append({
                "name": str(cam),
                "time": date_str,
                "gps": f"{cam.latitude},{cam.longitude}",
            })
        return route_data
