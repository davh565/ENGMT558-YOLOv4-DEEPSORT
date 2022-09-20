#================================================================
#
#   File name   : object_tracker.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : code to track detected object from video or webcam
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time
import pandas as pd
import pytesseract
import imutils
import math as m

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

file_str = "Administrator_IPCamera 08_20220617102156_20220617114348 2"

video_path   = "./IMAGES/carpark3.mp4"
crop_path = "./cropped_detections"
crop_threshold = 0.6 #min confidence required to crop detection
def Object_tracking(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = [],max_age=30, n_init=3):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance=0.6, max_age=max_age, n_init=n_init)

    times, times_2 = [], []

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())
    
    mask = np.zeros((height,width),"uint8")
    mask = cv2.rectangle(mask, (125, 345), (1750, 1080), 255, -1)
    frame_no = 0
    # Obtain info from the tracks
    tracked_bboxes = []
    paths = []
    best_confidences = []
    seconds = []
    ids = []
    frames = []
    timestamps = []
    while True:
        _, frame = vid.read()
        frame_no = frame_no + 1

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        frame_masked = cv2.bitwise_and(original_frame, original_frame, mask=mask)
        # cv2.imwrite("./mask.png", frame_masked)
        
        image_data = image_preprocess(np.copy(frame_masked), [input_size, input_size])
        #image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        #t1 = time.time()
        #pred_bbox = Yolo.predict(image_data)
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
            ####################################################################
            if track.confidence > track.best_confidence:
                track.best_confidence = track.confidence
                if track.best_confidence >= crop_threshold:
                    bbox_int = [int(x) for x in bbox]
                    bbox_int = [0 if x < 0 else x for x in bbox_int]
                    cropped_obj = frame[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
                    # cropped_obj = frame[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
                    # construct image name and join it to path for saving crop properly
                    obj_name = 'vehicle_' + str(track.track_id) + '.png'
                    obj_path = os.path.join(crop_path, obj_name )
                    # save image
                    paths.append(obj_path)
                    frames.append(frame_no)
                    best_confidences.append(track.best_confidence)
                    seconds.append(frame_no/fps)
                    ids.append(track.track_id)
                    timestamps.append(get_timestamp(file_str, fps, frame_no))
                    print(timestamps[-1])
                    cv2.imwrite(obj_path, cropped_obj)
                    
                    print("{:.2f}".format(frame_no/18), [int(x) for x in bbox],"{:.2f}".format(track.best_confidence), obj_name)
                

            ####################################################################
        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=CLASSES, tracking=True, rectangle_colors=rectangle_colors)

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        
        # image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

        # print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        # if show:
        #     cv2.imshow('output', image)
            
        #     if cv2.waitKey(25) & 0xFF == ord("q"):
        #         cv2.destroyAllWindows()
        #         break
            
    # cv2.destroyAllWindows()
    df = pd.DataFrame({ 'id': ids,
                        'frame': frames,
                        'timestamp': timestamps,
                        'path': paths,
                        'confidence': best_confidences})
    print(df)
    df = df.drop_duplicates('id' , keep='last').sort_values('id' , ascending=True).reset_index(drop=True)
    df.to_csv("./detections.csv", index=False)
    return df


def detect_plate(Yolo,df ,image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)
        
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    # image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
    # cropped_obj = frame[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    # construct image name and join it to path for saving crop properly

    # save image
    # df = pd.read_csv("./detections.csv")
    # df.set_index("path", inplace=True)
    for bbox in bboxes:
        if bbox[5] == 0:
            bbox_int = [int(x) for x in bbox]
            bbox_int = [0 if x < 0 else x for x in bbox_int]
            cropped_obj = original_image[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
            cropped_obj = cv2.cvtColor(cropped_obj, cv2.COLOR_BGR2GRAY)
            cropped_obj = cv2.resize( cropped_obj, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
            # perform otsu thresh (using binary inverse since opencv contours work better with white text)
            ret, thresh = cv2.threshold(cropped_obj, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kern)
            
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rect_kern)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, rect_kern)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours = imutils.grab_contours(contours)
            mask = np.ones(thresh.shape[:2], dtype="uint8") * 255
            # loop over the contours
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                # if the contour is bad, draw it on the mask
                if w >= 0.6*thresh.shape[1] or h >= thresh.shape[0] or cv2.contourArea(c) <= 0.01*thresh.shape[0]*thresh.shape[1]:
                    cv2.drawContours(mask, [c], -1, 0, -1)
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
            thresh = cv2.dilate(thresh, rect_kern, iterations = 1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rect_kern)
            rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kern)
            thresh = cv2.GaussianBlur(thresh, (5,5), 0)
            thresh = cv2.medianBlur(thresh, 3)
            thresh = cv2.bitwise_not(thresh)
            # area = cv2.contourArea(contours[0])
            # print(area)
            # contours = np.array(contours).reshape((-1,1,2)).astype(np.int32)
            # thresh = cv2.drawContours(thresh, contours, 3, (0,255,0), 3)
            plate_str = pytesseract.image_to_string(thresh).strip()
            plate_str = ''.join(c for c in plate_str if c.isalnum())
            # print(plate_str)
            df.loc[df['path'] == image_path.rsplit('/', 1)[1],["plate"]] = plate_str
            cv2.imwrite(output_path, thresh)
    return df
    # print(df)
    # # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

    # if output_path != '': cv2.imwrite(output_path, image)
    # if show:
    #     # Show the image
    #     # cv2.imshow("predicted image", image)
    #     # Load and hold the image
    #     cv2.waitKey(0)
    #     # To close the window after the required kill value was provided
    #     cv2.destroyAllWindows()
        
    # return image

def get_timestamp(filename, fps, frame_num):
    date_str = filename.split("_")[2]
    init_year, init_month, init_day, init_hour, init_min, init_sec = int(date_str[0:4]),int(date_str[4:6]),int(date_str[6:8]),int(date_str[8:10]),int(date_str[10:12]),int(date_str[12:14])
    hours = (fps / frame_num)/3600
    # hours_int = m.trunc(hours)
    hours_rem = hours - int(hours)
    minutes = hours_rem * 60
    # minutes_int = m.trunc(minutes)
    minutes_rem = minutes - int(minutes)
    seconds = int(minutes_rem * 60)
    # seconds_int = round(seconds)
    SS = init_sec + seconds
    MM = init_min + int(minutes)
    HH = init_hour + int(hours)
    dd = init_day
    mm = init_month
    yy = init_year
    if SS > 60:
        SS = SS - 60
        MM = MM + 1
    if MM > 60:
        MM = MM - 60
        HH = HH + 1
    if HH > 24:
        HH = HH - 24
        dd = dd + 1
    return  str(HH) + ":"+ str(MM) +":"+ str(SS)+"-"+str(yy) +"_"+ "{:02d}".format(str(mm)) +"_"+ "{:02d}".format(str(dd))
    
yolo = Load_Yolo_model()
df = Object_tracking(yolo,
                video_path,
                "detection.mp4",
                input_size=YOLO_INPUT_SIZE,
                show=False, 
                iou_threshold=0.1,
                rectangle_colors=(0,0,255),
                Track_only = ["Car", "Truck", "Bus", "Van"],
                n_init = 12,
                max_age=15)

df = pd.read_csv("./detections.csv")
df['plate'] = "<no plate>"

for path in df.path.iteritems():
    df = detect_plate(yolo,df, "cropped_detections/"+path[1], "plates/"+path[1], input_size=YOLO_INPUT_SIZE, show=False, CLASSES=YOLO_COCO_CLASSES, rectangle_colors=(255,0,0),score_threshold=0.1, iou_threshold=0.2)
df.plate = df.plate.replace("","<not readable>")
df['colour'] = "<none>"
print(df)