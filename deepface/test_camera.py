#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 04:13:10 2021

@author: erik.edmonds
"""
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
import re

from deepface.detectors import FaceDetector
from deepface import DeepFace
from deepface.extendedmodels import Age
from deepface.commons import functions, realtime, distance as dst
#------------------------
 

def operation(db_path=''):  
    face_detector = FaceDetector.build_model('opencv')
    print("Detector backend is ", 'opencv')
       
       	#------------------------
       
    input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
       
    text_color = (255,255,255)
       
    employees = []
       	#check passed db folder exists
    if os.path.isdir(db_path) == True:
        for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
            for file in f:
                if ('.jpg' in file):
                    exact_path = r + "/" + file
                    employees.append(exact_path)
    
    if len(employees) == 0:
        print("WARNING: There is no image in this path ( ", db_path,") . Face recognition will not be performed.")
    
    	#------------------------
    
    if len(employees) > 0:
    
        model = DeepFace.build_model('VGG-Face')
        print('VGG-Face'," is built")
    
    		#------------------------
    
        input_shape = functions.find_input_shape(model)
        input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
    
    		
        threshold = dst.findThreshold('VGG-Face', 'VGG-Face')
    
    	#------------------------
    	#facial attribute analysis models
    tic = time.time()
    
    emotion_model = DeepFace.build_model('Emotion')
    print("Emotion model loaded")
    
    age_model = DeepFace.build_model('Age')
    print("Age model loaded")
    
    gender_model = DeepFace.build_model('Gender')
    print("Gender model loaded")
    race_model = DeepFace.build_model('Race')
    print("Race model loaded")
    toc = time.time()
    
    print("Facial attibute analysis models loaded in ",toc-tic," seconds")
    
    	#------------------------
    
    	#find embeddings for employee list
    
    tic = time.time()
    
    	#-----------------------
    
    pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')
    
    	#TODO: why don't you store those embeddings in a pickle file similar to find function?
    
    embeddings = []
    	#for employee in employees:
    for index in pbar:
        employee = employees[index]
        pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
        embedding = []
    
    		
        img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend= 'opencv')
        img_representation = model.predict(img)[0,:]
    
        embedding.append(employee)
        embedding.append(img_representation)
        embeddings.append(embedding)
    
    df = pd.DataFrame(embeddings, columns = ['employee', 'embedding'])
    df['VGG-Face'] = 'VGG-Face'
    
    toc = time.time()
    
    print("Embeddings found for given data set in ", toc-tic," seconds")
    
    	#-----------------------
    
    pivot_img_size = 112 #face recognition result image
    
    	#-----------------------
    
    freeze = False
    face_detected = False
    face_included_frames = 0 
    freezed_frame = 0
    tic = time.time()
    
    cap = cv2.VideoCapture(0) #webcam
    
    while(True):
        ret, img = cap.read()
    
        if img is None:
            break
    
        raw_img = img.copy()
        resolution = img.shape; resolution_x = img.shape[1]; resolution_y = img.shape[0]
    
        if freeze == False:
            faces = FaceDetector.detect_faces(face_detector, 'opencv', img, align = False)
    
            if len(faces) == 0:
                face_included_frames = 0
        else:
            faces = []
    
        detected_faces = []
        face_index = 0
        for face, (x, y, w, h) in faces:
            if w > 130:
                face_detected = True
                if face_index == 0:
                    face_included_frames = face_included_frames + 1 #increase frame for a single face
    
                cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
    
                cv2.putText(img, str(5 - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
    
                detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
    
    				#-------------------------------------
    
                detected_faces.append((x,y,w,h))
                face_index = face_index + 1
    
    				#-------------------------------------
    
        if face_detected == True and face_included_frames == 5 and freeze == False:
            freeze = True
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()
    
        if freeze == True:
    
            toc = time.time()
            if (toc - tic) < 5:
    
                if freezed_frame == 0:
                    freeze_img = base_img.copy()
    					#freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue
    
                    for detected_face in detected_faces_final:
                        x = detected_face[0]; y = detected_face[1]
                        w = detected_face[2]; h = detected_face[3]
    
                        cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
    
    						#-------------------------------
    
    						
                        custom_face = base_img[y:y+h, x:x+w]