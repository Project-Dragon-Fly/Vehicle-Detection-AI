import argparse
import darknet
import time
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
        path = path.replace("\ ", " ")
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
        self.video_fno = 0
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
                self.video_fno += 1
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
        if self.file_idx > self.file_cnt or self.file_idx < 0:
            return None, None
        if self.file_idx == self.file_cnt:  # the last file
        	filename = os.path.basename(self.file[self.file_idx-1])
        else:
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
                            default="/home/edwin/Programing/DragonFly/Vehicle-Detection-AI/backup/yolov4-vehicle_best.weights",
                            help="path to YOLOv4 weights")
        parser.add_argument("--config_file", type=str,
                            default="/home/edwin/Programing/DragonFly/Vehicle-Detection-AI/cfg/yolov4-vehicle.cfg",
                            help="path to YOLOv4 config file")
        parser.add_argument("--data_file", type=str,
                            default="/home/edwin/Programing/DragonFly/Vehicle-Detection-AI/data/yolov4-vehicle.data",
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

    def save_frame(self, image):
        name, ext = self.frame_set.get_filename_type()
        if Frame_Parser.is_image_file(name+ext):
            path = os.path.join(self.args.media_out, name+"_detect"+ext)
            path = os.path.abspath(path)
            print("saving image to", path)

        elif Frame_Parser.is_video_file(name+ext):
            dir_path = os.path.join(self.args.media_out, name+"_detect_frames")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            path = os.path.join(dir_path, str(self.frame_set.video_fno)+'.png')
            path = os.path.abspath(path)
            print("saving frame to", path)

        cv2.imwrite(path, image)

    def save_label(self, detections, class_names, height, width):
        label_map = {k[1]: int(k[0]) for k in enumerate(class_names)}

        name, ext = self.frame_set.get_filename_type()
        if Frame_Parser.is_image_file(name+ext):
            path = os.path.join(self.args.label_out, name+".txt")
            path = os.path.abspath(path)
            print("saving image label to", path)

        elif Frame_Parser.is_video_file(name+ext):
            dir_path = os.path.join(self.args.label_out, name+"_detect_frames")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            path = os.path.join(dir_path, str(self.frame_set.video_fno)+'.txt')
            path = os.path.abspath(path)
            print("saving frame labels to", path)

        with open(path, 'a+') as f:
            for label, confidence, bbox in detections:
                x, y, w, h = bbox
                x /= width
                w /= width
                y /= height
                h /= height
                print("%d :%s@%.2f %.4f %.4f %.4f %.4f" %
                      (label_map[label], label, float(confidence), x, y, w, h))
                print("%d %.4f %.4f %.4f %.4f" %
                      (label_map[label], x, y, w, h), file=f)


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
    stat, frame = frame_set.next()
    prev_time = time.time()

    while stat:

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height))
        darknet_image = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image,
                                          thresh=cmd.args.thresh)
        darknet.free_image(darknet_image)

        if cmd.args.media_out is not None:
            image = darknet.draw_boxes(detections, image_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (frame.shape[1], frame.shape[0]))
            cmd.save_frame(image)
        if cmd.args.label_out is not None:
            cmd.save_label(detections, class_names, height, width)

        fps = (1/(time.time()-prev_time))
        prev_time = time.time()
        print("%2.4f completed. Current FPS: %.4f" %
              (frame_set.file_idx*100/frame_set.file_cnt, fps))

        stat, frame = frame_set.next()


main()
