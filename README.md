### live_Object_Detection (OpenCV)

### overview :

Embark on a tech journey with our captivating project – a live object detection (opencv) marvel! Utilizing OpenCV and MobileNetSSD, this code transforms your laptop camera or webcam into a vigilant eye. It tirelessly scans each video frame, identifying objects like people, chairs, and dogs, enclosing them in a box. Dive into the beauty of code and witness the magic unfold! 
Here, we will go through the steps required for setting up the project and some explanation about the code.

## Demo :

    

# Prologue :

By these following steps u can build a project on ur own :

1 .Install Dependencies: Ensure OpenCV and MobileNetSSD are  installed.
2 .Set Up Camera: Connect your laptop camera or webcam.
3 .Code Exploration: Dive into the code, where each frame is meticulously analyzed.
4 .Object Detection: Watch as the algorithm identifies persons, chairs, dogs, and more.
5 .Bounding Boxes: Objects are elegantly framed in boxes for clear visualization.

### To Clone /Run the Existing Project Follow below Steps :

**Step 1:** Create a directory in your local machine and cd into it

   ```
   mkdir ~/Desktop/opencv_project
   cd ~/Desktop/opencv_project

   ```

**Step 2:** Clone the repository and cd into the folder:

```
git clone https://github.com/navyasweet/live-Object-Detection (OpenCV).git
cd live-Object-Detection (OpenCV)
```
**Step 3:** Install all the necessary libraries. I used MacOS for this project. These are some of the libraries I had to install:

```
brew install opencv
pip install opencv-python
pip install opencv-contrib-python
pip install opencv-python-headless
pip install opencv-contrib-python-headless
pip install matplotlib
pip install imutils
```

Make sure to download and install opencv and and opencv-contrib releases for OpenCV  4.8. This ensures that the deep neural network (dnn) module is installed. You must have OpenCV 4.8 (or newer) to run this code.

**Step 4:** Make sure you have your video devices connected (e.g. Webcam, FaceTime HD Camera, etc.). You can list them by typing this in your terminal

```
system_profiler SPCameraDataType
system_profiler SPCameraDataType | grep "^    [^ ]" | sed "s/    //" | sed "s/://"
```

**Step 5:** To start your video stream and live object detection, run the following command:

```
python real_time_object_detection.py -p C:\Users\navya\OneDrive\Documents\docs\dronix\live-Object-Detection (OpenCV)-master\MobileNetSSD_deploy.prototxt.txt -m C:\Users\navya\OneDrive\Documents\docs\dronix\live-Object-Detection (OpenCV)-master\MobileNetSSD_deploy.caffemodel

```
## Technology uaed :
    
![python-libraries-for-ML](https://github.com/navyasweet/Live-Object-Detection-OPENCV-/assets/134292286/ac68319f-9db4-4e77-a234-3bb414d32941)
