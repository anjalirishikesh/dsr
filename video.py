

import random
import re
import os
import tempfile
import ssl
import cv2
import scipy.special
import numpy as np
import pickle as pkl
import math
import copy
import tensorflow as tf
import os
from urllib import request  # requires python3


interpreter = tf.lite.Interpreter(model_path="onnx_lite64.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_label():
  with open("labels.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    return labels

# load the label
word_data = load_label(
    
  )
# load the tensorflow lite run time


def sign(path,output):
    if os.path.isfile("frames.txt"):
        print("file exists")
    else:
        file  = open("frames.txt","w")
    # resize
    resize = (224,224)
    path_input = path
    # read the video
    cap = cv2.VideoCapture(path_input)
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("the total frames",length)

    # Get the Default resolutions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and filename.
    out = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
    frames = []

    in_frames = 65
    counter = 1
    frame_counter = 1
    try:
        while True:
            ret, frame_old = cap.read()
            if not ret:
                break
            frame_new = copy.deepcopy(frame_old)
            frame = crop_center_square(frame_old)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]] 
            frames.append(frame)
            # check the counter is not zero and append the frames
            if counter!=0:
                if counter % in_frames == 0:
                    
                    del frames[:]
            else:
                frames.append(frame)
            res = np.array(frames) / 255.0
            print(res.shape)
            value = res.shape[0] 
            counter+=1
            # Remove the frame with zero clip
            if value !=0 and value == 64:
                file  = open("frames.txt","w").write(str(frame_counter))
                frame_counter+=1
                
                #print(res.shape)
                model_input = tf.constant(res, dtype=tf.float32)[tf.newaxis, ...]
                print(model_input.shape)
                inp = tf.transpose(model_input, perm=[0, 4, 1, 2,3])
                print(inp.shape)
                interpreter.set_tensor(input_details[0]['index'], inp)
                interpreter.invoke()
                outputs = interpreter.get_tensor(output_details[0]['index'])
                #outputs = model(inp)
                topk=1
                result = []
                num_clips =1
                num_detections = 2000
                for i in range(num_detections):

                    res = outputs[0][i]
                    result.append(res)
                #print(result)
                res = []
                res.append(result)
                
                raw_scores = np.array(res)

                prob_scores = scipy.special.softmax(raw_scores, axis=1)
                prob_sorted = np.sort(prob_scores, axis=1)[:, ::-1]
                pred_sorted = np.argsort(prob_scores, axis=1)[:, ::-1]
                



                word_topk = None
                for k in range(topk):

                    for i, p in enumerate(pred_sorted[:, k]):
                        if str(word_data[p])!="want":
                            print(str(word_data[p]))
                        
                            word_topk= word_data[p]
                prob_topk = prob_sorted[:, :topk].transpose()
                print("Predicted signs:")
                print(word_topk)
                print(type(prob_topk))


                cv2.putText(frame_new, str(word_topk),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                
                
                

            # All the results have been drawn on the frame, so it's time to display it.
            out.write(frame_new)
    finally:
        os.remove("frames.txt")
        cap.release()


