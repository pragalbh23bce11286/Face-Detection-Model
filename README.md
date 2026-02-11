# Face Detection Using OpenCV (Haar Cascade)

This project demonstrates a basic face detection system using OpenCV’s Haar Cascade Classifier. It detects human faces in a static image, draws bounding boxes around them, and displays the result using Matplotlib.

# Key Features

Detects frontal human faces from an image

Uses a pre-trained Haar Cascade model

Converts images to grayscale for faster and efficient detection

Draws bounding boxes around detected faces

Includes error handling for missing or invalid image files

# Technology Used

Python

OpenCV (cv2)

Matplotlib

# How It Works

The image is loaded using OpenCV (cv2.imread)

The image is converted to grayscale, which improves detection performance

A Haar Cascade frontal face classifier is loaded

Faces are detected using detectMultiScale() with optimized parameters:

scaleFactor controls image scaling

minNeighbors reduces false positives

minSize filters out very small faces

Rectangles are drawn around detected faces

The processed image is displayed using Matplotlib
