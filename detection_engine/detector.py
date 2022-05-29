import darknet
import cv2
import time


class Detector:

    def __init__(self, config_file, data_file, weights, batch_size, threshold=0.10):
        self.network, self.class_names, self.class_colors = darknet.load_network(
            config_file,
            data_file,
            weights,
            batch_size=batch_size
        )
        self.threshold = threshold
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)

    def detect_frame(self, frame):
        start_time = time.time()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        darknet_image = darknet.make_image(self.width, self.height, 3)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, darknet_image,
                                          thresh=self.threshold)
        darknet.free_image(darknet_image)

        result = self._process_detection(detections)
        fps = (1/(time.time()-start_time))

        return result, fps

    def detect_frame_set(self, frame_set):
        stat, frame = frame_set.next()

        while stat:
            result, fps = self.detect_frame(frame)
            print("%2.4f completed. Current FPS: %.4f" %
                  (frame_set.file_idx*100/frame_set.file_cnt, fps))
            # code to proccess result
            stat, frame = frame_set.next()

    def _process_detection(self, detections):
        class_names, height, width = self.class_names, self.height, self.width
        label_map = {k[1]: int(k[0]) for k in enumerate(class_names)}
        result = []

        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            x /= width
            w /= width
            y /= height
            h /= height

            result.append({
                "label": label_map[label],
                "confidence": float(confidence),
                "bbox": (x, y, w, h)
            })
            # print("%d %.4f %.4f %.4f %.4f" % (label_map[label], x, y, w, h))

        return result
