from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.config.parse_names("data/classes.names")
yolo.config.parse_cfg("data/pip-yolov4-vehicle.cfg")

yolo.make_model()
yolo.load_weights("data/yolov4-vehicle_best.weights", weights_type="yolov")
yolo.summary(summary_type="yolov")
yolo.summary()
yolo.model.save("data/vehicle",save_format="tf")
