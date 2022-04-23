from darknet import load_network, network_width, network_height, make_image
from darknet import copy_image_from_bytes, detect_image, free_image
from darknet import bbox2points
import cv2


cfg_path = "cfg/yolov4-vehicle.cfg"
data_path = "data/yolov4-vehicle.data"
weight_path = "backup/yolov4-vehicle_last.weights"
image_file = "../yolov4-dataset/camera-set1/v06frame-126009.jpg"


network, class_names, class_colors = load_network(
    cfg_path, data_path, weight_path)
width = network_width(network)
height = network_height(network)


# darknet helper function to run detection on image


def darknet_helper(img, width, height):
    darknet_image = make_image(width, height, 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    img_height, img_width, _ = img.shape
    width_ratio = img_width/width
    height_ratio = img_height/height

    copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image, 0.1)
    free_image(darknet_image)
    return detections, width_ratio, height_ratio


image = cv2.imread(image_file)
detections, width_ratio, height_ratio = darknet_helper(image, width, height)
box_dict = dict()

for label, confidence, bbox in detections:

    x, y = bbox[0], bbox[1]

    left, top, right, bottom = bbox2points(bbox)
    left = int(left*width_ratio)
    right = int(right*width_ratio)
    top = int(top*height_ratio)
    bottom = int(bottom * height_ratio)

    key = (left, top, right, bottom)
    if key not in box_dict:
        box_dict[key] = list()
    box_dict[key].append(f"{label}[{confidence}]")

    # print(label, confidence, "box:", key)
    cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
    text = f"{label} [{confidence}]"
    text_pos = (left, top - 5)
    cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, class_colors[label], 2)

cv2.imwrite("./predictions.jpg", image)

update = True
while update:
    update = False


for key in box_dict:
    print(key)
    for detect in box_dict[key]:
        print("\t", detect)
