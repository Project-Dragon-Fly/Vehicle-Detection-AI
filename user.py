from camera.camera import camera_obj
from AI_yolov4.model import get_valid_class_label
import pandas as pd


class User:
    def input(self):
        v_options = get_valid_class_label()
        v_type=""
        while v_type not in v_options:
            print("Enter vehicle type: ")
            print("options : ",v_options)
            v_type = input("v_type: ")
        
        c_options = { i:camera_obj[i].cam_name for i in camera_obj }
        c_id = ""
        while c_id not in c_options:
            print("Enter starting camera:")
            print("options : ",c_options)
            c_id = int(input("c_id: "))
        
        start_time = int(input("start time: "))
        end_time = int(input("end time: "))

        return v_type, camera_obj[c_id],start_time,end_time

    def message(self,string):
        print(string)

    def select_one_vehicle(self,vehicles):
        index = 0
        print("select any one vehicle \n")
        for vehi in vehicles:
            print(f"index : {index}")
            print(f"camera : {camera_obj[vehi['cam_id']]}")
            print(f"time : {vehi['entry_time']} to {vehi['exit_time']} seconds")
            print("-"*80)
            index += 1
            
        try:
            index = int(input(f"Enter the vehicle index [0-{index-1}]: "))
            return vehicles[index]
        except KeyboardInterrupt:
            return None
        


user = User()
v_type, camera, start_time, end_time = user.input()
vehicle_history = dict()

print("scanning initial camera",camera)
vehicles = camera.filter_frame(start_time,end_time,v_type)


if len(vehicles) == 0:
    user.message("no vehicles found")
    exit()
elif len(vehicles) > 1:
    vehicle = user.select_one_vehicle(vehicles)
    if vehicle is None:
        user.message("progam exit")
else:
    vehicle = vehicles[0]

    
for key in vehicle:
    vehicle_history[key] = list()
    vehicle_history[key].append(vehicle[key])


while True:

    cam_id = vehicle['cam_id']
    cam = camera_obj[cam_id]
    next_cam_dict = cam.next_camera()
    
    vehicles = list()
    for nxt_cam_id in next_cam_dict:
        nxt_cam = camera_obj[nxt_cam_id]
        min_time = next_cam_dict[nxt_cam_id]['min_time'] + vehicle['exit_time']
        max_time = next_cam_dict[nxt_cam_id]['max_time'] + vehicle['exit_time']
        
        print(f"scanning camera {nxt_cam} from {min_time} to {max_time} seconds")
        vehicles.extend(nxt_cam.filter_frame(min_time,max_time,v_type))

    if len(vehicles) == 0:
        user.message("Terminating the search")
        break
    elif len(vehicles) > 1:
        vehicle = user.select_one_vehicle(vehicles)
        if vehicle is None:
            user.message("Terminating the search")
            break
    else:
        vehicle = vehicles[0]
    
    for key in vehicle:
        vehicle_history[key].append(vehicle[key])


df = pd.DataFrame(vehicle_history)
df = df.sort_values('exit_time')

print("Travelled route")
for index,row in df.iterrows():
    cam = camera_obj[row['cam_id']]
    print(f"{cam} from {row['entry_time']} to {row['exit_time']} at {cam.latitude},{cam.longitude}")
