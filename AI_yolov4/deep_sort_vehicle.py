import numpy as np
from AI_yolov4.model import predict as yolo_predict
from AI_yolov4.deep_sort import preprocessing, nn_matching
from AI_yolov4.deep_sort.detection import Detection
from AI_yolov4.deep_sort.tracker import Tracker
from AI_yolov4.tools import generate_detections as gdet

CONSTS = dict()
CONSTS['deep_sort_model'] =  'AI_yolov4/data/mars-small128.pb'
CONSTS['max_cosine_distance'] = 0.4
CONSTS['nn_budget'] = None
CONSTS['nms_max_overlap'] = 1.0
encoder = gdet.create_box_encoder(CONSTS['deep_sort_model'], batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", CONSTS['max_cosine_distance'], CONSTS['nn_budget'])

# ------------------------------------------------------------------------------
class Vehicle:
	def __init__(self,type):
		self.type = type
		self.last_seen = list()
		self.features = None

	def find_in_camera(self,camera, start_time, end_time):
		tracker = Tracker(metric)
		valid_targets = dict()

		frame_no = -1
		for frame in camera.filter_frame(start_time,end_time):
			names, bboxes, scores = yolo_predict(frame)
			frame_no += 1

			i = 0
			while i < len(names):
				if names[i] != self.type:
					names = np.delete(names,i,axis=0)
					bboxes = np.delete(bboxes,i,axis=0)
					scores = np.delete(scores,i,axis=0)
				else:
					i += 1

			frame_size = frame.shape[:2]

			#change bboxes to x_min,y_min,width,height
			bboxes[:,0] -= bboxes[:,2]/2  # x_min
			bboxes[:,1] -= bboxes[:,3]/2  # y_min
			bboxes[:,0] *= frame_size[1]
			bboxes[:,2] *= frame_size[1]
			bboxes[:,1] *= frame_size[0]
			bboxes[:,3] *= frame_size[0]

			# encode yolo detections and feed to tracker
			features = encoder(frame, bboxes)
			detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

			# run non-maxima supression
			boxs = np.array([d.tlwh for d in detections])
			scores = np.array([d.confidence for d in detections])
			classes = np.array([d.class_name for d in detections])
			indices = preprocessing.non_max_suppression(boxs, classes, CONSTS['nms_max_overlap'], scores)
			detections = [detections[i] for i in indices]

			# Call the tracker
			tracker.predict()
			tracker.update(detections)

			# update tracks
			for track in tracker.tracks:
				if not track.is_confirmed() or track.time_since_update > 1:
					continue
				id = track.track_id
				bbox = track.to_tlbr()
				features = track.features

				if id not in valid_targets:
					valid_targets[id] = dict()
					valid_targets[id]['camera'] = camera
					valid_targets[id]['entry_time'] = start_time + frame_no/camera.fps
					valid_targets[id]['tlbr'] = list()
					valid_targets[id]['features'] = list()

				valid_targets[id]['exit_time'] = start_time + frame_no/camera.fps
				valid_targets[id]['tlbr'].append(bbox)
				valid_targets[id]['features'].append(features)

		valid_targets = [valid_targets[id] for id in valid_targets if valid_targets[id]['exit_time'] - valid_targets['start_time'] >= 1]
		return valid_targets

	def update_location(self,vehi):
		if self.features is None:
			self.features = vehi['features']
		else:
			self.features.extend(vehi['features'])
		self.last_seen.append([vehi['camera'], vehi['entry_time'], vehi['exit_time']])

	def predict_next_cam(self):
		try:
			recent = self.last_seen[-1]
			recent_cam = recent['camera']
			recent_exit = recent['exit_time']

			predict = list()
			lnk = recent_cam.next_camera()
			for next_cam_id in lnk:
				next_cam = lnk[next_cam_id]['camera']
				min_time = recent_exit + lnk[next_cam_id]['min_time']
				max_time = recent_exit + lnk[next_cam_id]['max_time']
				predict.append((next_cam, min_time, max_time))
			return predict

		except Exception as e:
			print("No recent camera found, index error")
			raise e
