# A Fully Customizable Face Gesture Recognition System
Recognize two types of face gestures using Python and Mediapipe Framework. This work is inspired by
 [hand gesture recognition](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)

# Requirements
See `requirements.txt` for details. <br>
Use `pip install requirements.txt` to install required packages.

# Demo
Here's how to run the demo using your webcam.
```bash
python app.py
```

The following options can be specified when running the demo.
* --device<br>Specifying the camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：960)
* --height<br>Height at the time of camera capture (Default：540)
* --use_static_image_mode<br>Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
* --min_detection_confidence<br>
Detection confidence threshold (Default：0.5)
* --min_tracking_confidence<br>
Tracking confidence threshold (Default：0.5)

# Directory
<pre>
│  app.py
│  static_classification.ipynb
│  gesture_classification.ipynb
│  
├─model
│  ├─static_classifier
│  │  │  static_face.csv
│  │  │  static_classifier.hdf5
│  │  │  static_classifier.py
│  │  │  static_classifier.tflite
│  │  └─ static_classifier_label.csv
│  │          
│  └─gesture_classifier
│      │  gesture_face.csv
│      │  gesture_classifier.hdf5
│      │  gesture_classifier.py
│      └─ gesture_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>

### app.py
Main program to collect data and make real time inferences.

### static_classification.ipynb
Scripts for training static gesture recognition.

### gesture_classification.ipynb
Scripts for training dynamic gesture recognition.

### model/static_classifier
This directory stores files related to static gesture recognition.<br>
The following files are stored.
* Training data(static_face.csv)
* Trained model(static_classifier.tflite)
* Label data(static_classifier_label.csv)
* Inference module(static_classifier.py)

### model/gesture_classifier
This directory stores files related to dynamic gesture recognition.<br>
The following files are stored.
* Training data(gesture_face.csv)
* Label data(gesture_label.csv)
* Inference module(gesture_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.

# How to use `app.py`
Default to key `N` where static inference will be made. `K` and `L` are static and dynamic inferences mode to collect data. Refer to [Training](#training). `X` key to make inferencess on dynamic gestures. 

# Training
All gestures can be modified and created at will. To apply changes, make sure to follow the steps and retrain the model.

### Static gesture training
#### 1.Learning data collection
Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）<br>
If you press "0" to "9", the data will be added to `model/static_classifier/static_face.csv` <br>

#### 2.Model training
Run all cells in `static_classifier.ipynb`.

### Dynamic gesture traning
#### 1. Learning data collection
Press "l" to enter the mode to save key points（displayed as 「MODE:Logging Point History」）<br>
If you press "0" to "9", the data will be added to `model/gesture_classifier/gesture_face.csv`. In this mode, depends on the system, you can change the length of captured `history_length` in `app.py` to capture more or less points history. When the number is pressed, it will capture previous points history of specified length. <br>

#### 2. Model training
Run all cells in `gesture_classifier.ipynb`.

#### 3. Model inference
Press "x" to enter the mode to save key points（displayed as 「MODE:Gesture Inference」

# Reference
* [MediaPipe](https://mediapipe.dev/)
* [Kazuhito's hand gesture recognition](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)