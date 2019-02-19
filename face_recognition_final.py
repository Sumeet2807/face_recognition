import cv2
import sys
import os
import numpy as np
import argparse
import face_recognition
import face_recognition_knn as knn
# Get user supplied values

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
# ap.add_argument("-o", "--output", required=True,
#     help="path to output image")
ap.add_argument("-p", "--prototxt", default="CaffeModel/deploy.prototxt.txt",
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="CaffeModel/res10_300x300_ssd_iter_140000.caffemodel",
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-pm", "--prediction_model", default="predict_model/trained_knn_model.clf",
    help="path to input model")
args = vars(ap.parse_args())




img_src = args["image"]
# pathf = args["output"]
# pathf = pathf + "/"

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("detection Model Loaded")

for file in os.listdir(img_src):
    
    if file.endswith(".jpg"):                

        
        imagePath = img_src + "/" + file


# Read the image
        image = cv2.imread(imagePath)
        (h, w) = image.shape[:2]
        print("detecting face")
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()  
        print("face detected")
        face_locations = []

        for i in range(0, detections.shape[2]):



    
            confidence = detections[0, 0, i, 2]
 
            if confidence > args["confidence"]:
      
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                face_locations.append((startY, endX, endY, startX))

        print("recognizing faces")
        print(file)

        predictions = knn.predict(imagePath, face_locations, model_path=args["prediction_model"])
        for name, (top, right, bottom, left) in predictions:
        	print("- Found {} at ({}, {})".format(name, left, top))

                


 
  