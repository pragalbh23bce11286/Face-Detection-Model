# Face Detection using OpenCV
Project Overview:

This project implements real-time face detection using OpenCV's Haar Cascade Classifier. The system detects human faces in an image and highlights them with bounding boxes.

# Features

Detects multiple faces in a single image

Uses pre-trained Haar Cascade model

Converts image to grayscale for efficient detection

Displays face count on output image

Clean visualization using Matplotlib

# Technologies Used

Python

OpenCV

Matplotlib

Haar Cascade Classifier

# Project Structure
Face_Detection_Project/
│
├── face_detection.py
├── haarcascade_frontalface_default.xml
├── images/
│   └── friends.jpg
└── README.md

# Installation
pip install opencv-python matplotlib

# How to Run

Update image path:

image_path = "path_to_your_image.jpg"

Run the script:

python face_detection.py

# How It Works

Image is loaded using OpenCV.

Converted to grayscale.

Haar cascade detects facial regions.

Bounding boxes are drawn.

Matplotlib displays result with face count.

# Improvements

Real-time webcam detection

Eye detection

Smile detection

Face recognition (using LBPH or Deep Learning)

Age & Gender detection

Model performance comparison

# Future Enhancements

Replace Haar Cascade with:

MTCNN

DNN-based face detector

YOLO-based face detection

Add accuracy benchmarking dataset

Deploy as a web app

# Applications

Attendance systems

Security surveillance

Social media tagging

Smart cameras

Access control systems
