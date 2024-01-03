# import packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

# Arguments used here:
# prototxt = MobileNetSSD_deploy.prototxt.txt (required)
# model = MobileNetSSD_deploy.caffemodel (required)
# confidence = 0.2 (default)

# SSD (Single Shot MultiBox Detector) is a popular algorithm in object detection
# It has no delegated region proposal network and predicts the boundary boxes and the classes directly from feature maps in one single pass
# To improve accuracy, SSD introduces: small convolutional filters to predict object classes and offsets to default boundary boxes
# Mobilenet is a convolution neural network used to produce high-level features

# SSD is designed for object detection in real-time
# The SSD object detection composes of 2 parts: Extract feature maps, and apply convolution filters to detect objects

# Let's start by initializing the list of the 21 class labels MobileNet SSD was trained to.
# Each prediction composes of a boundary box and 21 scores for each class (one extra class for no object),
# and we pick the highest score as the class for the bounded object
CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Assign random colors to each of the classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model
# The model from Caffe: MobileNetSSD_deploy.prototxt.txt; MobileNetSSD_deploy.caffemodel;
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# warm up the camera for a couple of seconds
time.sleep(2.0)

# FPS: used to compute the (approximate) frames per second
# Start the FPS timer
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    # vs is the VideoStream
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]

    # Resize each frame
    resized_image = cv2.resize(frame, (300, 300))

    # Creating the blob
    blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)

    # pass the blob through the network and obtain the predictions
    net.setInput(blob)
    predictions = net.forward()

    # loop over the predictions
    for i in np.arange(0, predictions.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = predictions[0, 0, i, 2]

        # Filter out predictions with confidence levels below the specified threshold
        if confidence > args["confidence"]:
            # extract the index of the class label from the 'predictions'
            idx = int(predictions[0, 0, i, 1])
            # then compute the (x, y)-coordinates of the bounding box for the object
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Get the label with the confidence score
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("Object detected: ", label)

            # Draw a rectangle around the object
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            # Put a text outside the rectangular detection
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show the output frame
    cv2.imshow("Frame", frame)

    # Press 'q' key to break the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# Stop the timer
fps.stop()

# Display FPS Information: Total Elapsed time and an approximate FPS over the entire video stream
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

# Destroy windows and cleanup
cv2.destroyAllWindows()
# Stop the video stream
vs.stop()
