from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True,
# 	help="path to output directory")
# args = vars(ap.parse_args())
name =input('Enter your name please: ')
output_dir = os.mkdir('dataset/{}/'.format(name))
created_dir = 'dataset/{}/'.format(name)
proto_path = "face_detection_model/deploy.prototxt"
caffe_path = 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'
confidence_input = 0.5

# output_path = input("Please enter your name:\n",)
net = cv2.dnn.readNetFromCaffe(proto_path,caffe_path)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)
 
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
   

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < confidence_input:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    #define the screen resulation
    screen_res = 1280, 720
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)

    #resized window width and height
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    # #cv2.WINDOW_NORMAL makes the output window resizealbe
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
 
    # #resize the window according to the screen resolution
    cv2.resizeWindow('Frame',window_width,window_height)
    # show the output frame
    cv2.imshow("Frame", frame)


    key = cv2.waitKey(1) & 0xFF

    # if the `k` key was pressed, write the *original* frame to disk
    # so we can later process it and use it for face recognition
    if key == ord("k"):
        if not os.path.exists(created_dir): 
            p = os.path.sep.join([output_dir, "{}.png".format(
                str(total).zfill(5))])
            cv2.imwrite(p, orig)
            total += 1
        else:
            p = os.path.sep.join([created_dir, "{}.png".format(
                str(total).zfill(5))])
            cv2.imwrite(p, orig)
            total += 1
    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
