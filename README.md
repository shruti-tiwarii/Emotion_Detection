# Emotion
This software recognizes human faces and their corresponding emotions from a video or webcam feed. Powered by OpenCV and Deep Learning.

![Demo](https://github.com/petercunha/Emotion/blob/master/demo/demo.gif?raw=true)


## Installation

Clone the repository:
```
git clone https://github.com/petercunha/Emotion.git
cd Emotion/
```

Install these dependencies with `pip3 install <module name>`
-	tensorflow
-	numpy
-	scipy
-	opencv-python
-	pillow
-	pandas
-	matplotlib
-	h5py
-	keras

Once the dependencies are installed, you can run the project.
`python3 emotions.py`


## To train new models for emotion classification

- Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Move the downloaded file to the datasets directory inside this repository.
- Untar the file:
`tar -xzf fer2013.tar`
- Run the train_emotion_classification.py file:
`python3 train_emotion_classifier.py`

