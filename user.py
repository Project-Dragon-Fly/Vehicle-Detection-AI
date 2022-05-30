from camera.camera import camera_obj
from AI_yolov4.deep_sort_vehicle import Vehicle

class User:
    def input(self):
        v_type = input("v_type: ")
        v_color = input("v_color: ")
        camera_id = int(input("camera id: "))
        start_time = int(input("start time: "))
        end_time = int(input("end time: "))

        return v_type, v_color, camera_obj[camera_id],start_time,end_time

    def message(self,string):
        print(string)

    def select_one_target(self,valid_targets):
        index = 0
        print("select any one vehicle \n")
        for vehi in valid_targets:
            print(index,vehi['camera'], "time from",vehi['entry_time'],"to",vehi['exit_time'])
            index += 1
        index = int(input(f"Enter the vehicle id [0-{index}]: "))
        return valid_targets[index]


user = User()
v_type, v_color, camera, start_time, end_time = user.input()
target_vehi = Vehicle(v_type)


valid_targets = target_vehi.find_in_camera(camera,start_time,end_time)

if len(valid_targets) == 0:
    user.message("no vehicles found")
    exit()
elif len(valid_targets) > 1:
    target = user.select_one_target(valid_targets)
else:
    target = valid_targets[0]
target_vehi.update_location(target)



while True:

    next_search_cam = target_vehi.predict_next_cam()
    valid_targets = list()

    for camera,start_time,end_time in next_search_cam:
        valid_targets.extend(target_vehi.find_in_camera(camera,start_time,end_time))

    if len(valid_targets) == 0:
        user.message("no vehicles found, search ended")
        break
    elif len(valid_targets) > 1:
        target = user.select_one_target(valid_targets)
    else:
        target = valid_targets[0]
    target_vehi.update_location(target)



print("Travelled route")
for vehi in target_vehi.last_seen:
	print("camera:", vehi['camera'])
	print("enter:", vehi['entry time'])
	print("exit:", vehi['exit time'])

	print(" NEXT-> ")
print("END")
