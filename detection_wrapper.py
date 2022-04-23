from threading import Thread
from queue import Queue
from ctypes import *
import numpy as np
import argparse
import darknet
import random
import time
import glob
import os
import cv2


class Frame_Parser:
    """
    Identify the source type to build and parse frame collection
    """

    def is_video_file(path):
        filename = os.path.basename(path)
        file, ext = os.path.splitext(filename)
        return ext in {'.mp4'}

    def is_image_file(path):
        filename = os.path.basename(path)
        file, ext = os.path.splitext(filename)
        return ext in {'.png', '.jpg'}

    def is_media_file(path):
        if Frame_Parser.is_image_file(path) or Frame_Parser.is_video_file(path):
            return True
        return False

    def find_source(self):
        path = self.arg_path.rstrip('*')
        path = os.path.abspath(path)
        _, ext = os.path.splitext(path)

        self.file = list()  # list of all source path

        if Frame_Parser.is_media_file(path):
            self.file.append(path)

        elif os.path.isdir(path):
            for f in os.listdir(path):
                if Frame_Parser.is_media_file(f):
                    self.file.append(os.path.join(path, f))

        elif ext == '.txt':
            with open(path, 'r') as txt_file:
                for f in txt_file.readlines():
                    fpath = f.strip('*')
                    fpath = os.path.abspath(fpath)
                    _, ext = os.path.splitext(fpath)

                    if Frame_Parser.is_media_file(fpath):
                        self.file.append(fpath)

                    elif os.path.isdir(fpath):
                        for ff in os.listdir(fpath):
                            if Frame_Parser.is_media_file(ff):
                                self.file.append(os.path.join(fpath, ff))
        else:
            raise ValueError(f"Undefined source {path}")

    def __init__(self, path):
        self.arg_path = path
        self.find_source()

        self.file_cnt = len(self.file)
        self.frame_cnt = 0

        # compute the total frames
        for file in self.file:
            if Frame_Parser.is_image_file(file):
                self.frame_cnt += 1
            else:  # is video file
                video = cv2.VideoCapture(file)
                self.frame_cnt += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                video.release()
        print(f"[info] found {self.frame_cnt} frames in {self.file_cnt} files")

        # set the initial frame source
        self.file_idx = -1
        self.move_next_frame()

    def move_next_frame(self):
        self.file_idx += 1
        self.isvideo = False
        self.isimage = False
        if self.file_idx < self.file_cnt:
            self.isvideo = Frame_Parser.is_video_file(self.file[self.file_idx])
            self.isimage = Frame_Parser.is_image_file(self.file[self.file_idx])
            if self.isvideo:
                self.video = cv2.VideoCapture(self.file[self.file_idx])

    def next(self):
        while self.isvideo:
            self.isvideo, frame = self.video.read()
            if self.isvideo:
                return True, frame
            else:
                self.video.release()
                self.move_next_frame()

        if self.isimage:
            frame = cv2.imread(self.file[self.file_idx])
            self.move_next_frame()
            return True, frame
        return False, None

    def get_filename_type(self):
        if self.file_idx >= self.file_cnt or self.file_idx < 0:
            return None, None
        filename = os.path.basename(self.file[self.file_idx])
        return os.path.splitext(filename)


class Argument_Handler:
    """
    Class that enforce the argument rules and values
    """

    def __init__(self):
        self.args = Argument_Handler.definitions()
        self.validation()

    def definitions():
        parser = argparse.ArgumentParser(description="YOLOv4 Detection Model")
        parser.add_argument("--input", type=str, default=None,
                            help="Path to the source. Accepted sources"
                            "\n - video : mp4"
                            "\n - image : jpg, png"
                            "\n - txt : text file containing path to image(s)"
                            "\n - folder : directory to frame source")
        parser.add_argument("--media-out", type=str, default=None,
                            help="Directory to save predicted frames. "
                            "Default: do not save")
        parser.add_argument("--label-out", type=str, default=None,
                            help="Directory to save labels YOLOv4 format. "
                            "Default: do not save")

        parser.add_argument("--weights", type=str,
                            default="./backup/yolov4-vehicle_best.weights",
                            help="path to YOLOv4 weights")
        parser.add_argument("--config_file", type=str,
                            default="./cfg/yolov4-vehicle.cfg",
                            help="path to YOLOv4 config file")
        parser.add_argument("--data_file", type=str,
                            default="./data/yolov4-vehicle.data",
                            help="path to YOLOV4 data file")

        parser.add_argument("--thresh", type=float, default=.10,
                            help="Set minimum threshold detection. Removes"
                            "detections with confidence below this value.")
        parser.add_argument("--batch_size", type=int, default=1,
                            help="Number of parallel image processing")

        return parser.parse_args()

    def validation(self):
        if not os.path.exists(self.args.weights):
            raise ValueError(
                f"Cannot find weights at {os.path.abspath(self.args.weights)}"
            )

        if not os.path.exists(self.args.config_file):
            raise ValueError(
                f"Cannot find config file at {os.path.abspath(self.args.config_file)}"
            )

        if not os.path.exists(self.args.data_file):
            raise ValueError(
                f"Cannot find data file at {os.path.abspath(self.args.data_file)}"
            )

        if not (0 < self.args.thresh <= 1):
            raise ValueError("threshold should be between 0 and 1 (exclusive)")

        if self.args.batch_size <= 0:
            raise ValueError("batch size should be above 0 ")
        elif self.args.batch_size > 8:
            print("[warning] Reduce batch size if ran into resource errors")

        self.frame_set = Frame_Parser(self.args.input)


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    _height = darknet_height
    _width = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x = int(x * image_w)
    orig_y = int(y * image_h)
    orig_width = int(w * image_w)
    orig_height = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left = int((x - w / 2.) * image_w)
    orig_right = int((x + w / 2.) * image_w)
    orig_top = int((y - h / 2.) * image_h)
    orig_bottom = int((y + h / 2.) * image_h)

    if (orig_left < 0):
        orig_left = 0
    if (orig_right > image_w - 1):
        orig_right = image_w - 1
    if (orig_top < 0):
        orig_top = 0
    if (orig_bottom > image_h - 1):
        orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(
            network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = (1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename,
                            (video_width, video_height))
    try:
        while cap.isOpened():
            frame = frame_queue.get()
            detections = detections_queue.get()
            fps = fps_queue.get()
            detections_adjusted = []
            if frame is not None:
                for label, confidence, bbox in detections:
                    bbox_adjusted = convert2original(frame, bbox)
                    detections_adjusted.append(
                        (str(label), confidence, bbox_adjusted))
                image = darknet.draw_boxes(
                    detections_adjusted, frame, class_colors)
                if not args.dont_show:
                    cv2.imshow('Inference', image)
                if args.out_filename is not None:
                    video.write(image)
                # if cv2.waitKey(int(fps)) == 27:
                #    break
    finally:
        cap.release()
        video.release()
        cv2.destroyAllWindows()


def sub_main():
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Thread(target=video_capture, args=(
        frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue,
           detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue,
           detections_queue, fps_queue)).start()


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(
        batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(
        darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(
        network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(
            predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx])
                   for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)


def main():
    cmd = Argument_Handler()

    # loading the model
    network, class_names, class_colors = darknet.load_network(
        cmd.args.config_file,
        cmd.args.data_file,
        cmd.args.weights,
        batch_size=cmd.args.batch_size
    )
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    frame_set = cmd.frame_set

    try:

        while True:
            # loop asking for new image paths if no list is given
            stat, frame = frame_set.next()
            if not stat:
                break

            prev_time = time.time()

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)
            darknet_image = darknet.make_image(width, height, 3)
            darknet.copy_image_from_bytes(
                darknet_image, image_resized.tobytes())
            detections = darknet.detect_image(network, class_names, darknet_image,
                                              thresh=cmd.args.thresh)
            darknet.free_image(darknet_image)
            fps = (1/(time.time()-prev_time))

            # image = darknet.draw_boxes(detections, image_resized, class_colors)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            print("%2.4f completed. Current FPS: %.4f" %
                  (frame_set.file_idx*100/frame_set.file_cnt, fps))
    finally:
        pass


main()
