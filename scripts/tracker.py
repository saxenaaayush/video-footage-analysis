import cv2
def run_tracking(person_detector, box_detector, tracker, cap, writer, save_metadata=False):
    frame_idx = 0
    metadata_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection using person detector
        results_person = person_detector(frame, verbose=False)[0]
        # Run detection using box detector
        results_box = box_detector(frame, verbose=False)[0]

        detections = []

        # Process person detections (assuming "person" class label)
        for box in results_person.boxes.data.tolist():
            x1, y1, x2, y2, score, cls_id = box
            label = results_person.names[int(cls_id)]
            if label.lower() == 'person' and score > 0.4:
                detections.append(([x1, y1, x2 - x1, y2 - y1], score, label))
                
        # Process box detections (assuming "box" is the class label; adjust accordingly)
        for box in results_box.boxes.data.tolist():
            x1, y1, x2, y2, score, cls_id = box
            label = results_box.names[int(cls_id)]
            if label.lower() == 'box' and score > 0.4:
                detections.append(([x1, y1, x2 - x1, y2 - y1], score, label))

        # Update tracker with the combined detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Annotate frame for each confirmed track
        for track in tracks:
            if not track.is_confirmed():
                continue
            x, y, w, h = track.to_ltrb()
            track_id = track.track_id
            # Use the label from the detection if available.
            cls = track.get_det_class() or 'unknown'
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)
            cv2.putText(frame, f'{cls}-{track_id}', (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if save_metadata:
                metadata_log.append({
                    'frame': frame_idx,
                    'track_id': track_id,
                    'label': cls,
                    'bbox': [x, y, w, h]
                })

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    return metadata_log

