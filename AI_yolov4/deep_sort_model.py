        frame_size = frame.shape[:2]


        start_time = time.time()



        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        scores = pred_bbox[:,5]
        names = np.array([yolo.config.names[int(i)] for i in pred_bbox[:,4]])
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]


        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if len(out_txt)==1:
                out_txt.append("Tracking ")
            out_txt.append(f"{track.track_id}-{track.class_name} ")

            # draw on the frame
            color = (255,0,0)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+ (len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)


        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        out_txt.insert(1,"FPS: %.2f"%fps)

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(result)

        variables['pred_bbox'].append(copy.deepcopy(pred_bbox))
        variables['detections'].append(copy.deepcopy(detections))
        variables['tracker.tracks'].append(copy.deepcopy(tracker.tracks))


        print(" ".join(out_txt))
finally:
    print("saving the out file")
    out.release()
    cv2.destroyAllWindows()

    with open("variables.dat",'wb') as var_file:
        pickle.dump(variables,var_file)
