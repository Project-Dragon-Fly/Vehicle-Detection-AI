from yolov4.tf import YOLOv4

CONSTS = dict()
CONSTS['cfg'] = 'AI_yolov4/data/pip-yolov4-vehicle.cfg'
CONSTS['names'] = 'AI_yolov4/data/classes.names'

print("initialsing yolo")
yolo = YOLOv4()
yolo.config.parse_names(CONSTS['names'])

def get_valid_class_label():
    return [ yolo.config.names[i] for i in yolo.config.names]
