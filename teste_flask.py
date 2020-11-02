# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:57:44 2019
@author: seraj
"""
import time
import cv2 
from flask import Flask, render_template, Response
import os 
import face_recognition 


app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/hunt')
def hunt():
    """Video streaming home page."""
    return render_template('hunt.html')

def gen():
    """Video streaming generator function."""
    #cap = cv2.VideoCapture('768x576.avi')
    cap = cv2.VideoCapture(0)
    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret: #if vid finish repeat
            frame = cap = cv2.VideoCapture(0) #cv2.VideoCapture("768x576.avi")
            continue
        if ret:  # if there is a frame continue with code
            image = cv2.resize(frame, (0, 0), None, 1, 1)  # resize image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
            fgmask = sub.apply(gray)  # uses the background subtraction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(opening, kernel)
            retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            minarea = 400
            maxarea = 50000
            for i in range(len(contours)):  # cycles through all contours in current frame
                if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                    area = cv2.contourArea(contours[i])  # area of contour
                    if minarea < area < maxarea:  # area threshold for contour
                        # calculating centroids of contours
                        cnt = contours[i]
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # gets bounding points of contour to create rectangle
                        # x,y is top left corner and w,h is width and height
                        x, y, w, h = cv2.boundingRect(cnt)
                        # creates a rectangle around contour
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Prints centroid text in order to double check later on
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
                        cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,line_type=cv2.LINE_8)
        #cv2.imshow("countours", image)
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
           break
   
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
#--------------------
# Object Detection  |
#--------------------
thres = 0.45

# Creation de List Vide d'init
classNames = []
classFile = './lib/coco.names'
# ouverture fichier 
with open (classFile , 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

# creation des path et import pour visualisation ML
configPath = "./lib/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = './lib/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5,127.5))
net.setInputSwapRB(True)

@app.route('/teste')
def teste():
    """Video streaming home page."""
    return render_template('vid_obj.html')

def gentest(ret=True):
    """Video streaming generator function."""
    #cap = cv2.VideoCapture('768x576.avi')
    cap = cv2.VideoCapture(0)
    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret: #if vid finish repeat
            frame = cv2.VideoCapture("768x576.avi")
            continue
        if ret:  # if there is a frame continue with code
            image = cv2.resize(frame, (0, 0), None, 1, 1)  # resize image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
            fgmask = sub.apply(gray)  # uses the background subtraction

        classIds , confs, bbox = net.detect(image, confThreshold=0.5)
        #print(classIds, bbox)

        if len(classIds) !=0:
            # configuration affichage rectangle dans l'image
            for classIds , confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                # Creation du rectangle et definition couleur epaisseur de traits
                cv2.rectangle(image,box, color=(0,255,0), thickness=1) 
                # creation du texte label avec definition de sont emplacement dans la page 
                # comme la creation graphique sous word ou autre
                cv2.putText(image, classNames[classIds-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)


        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
           break
   
@app.route('/video_test')
def video_teste():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gentest(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
"""
#--------------------
# ID Detection  |
#--------------------
KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
video = cv2.VideoCapture(0)


def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)
"""
@app.route('/ID_people')
def ID_p():
    """Video streaming home page."""
    return render_template('vid_id.html')

def gen_id():
    """Video streaming generator function."""
    #cap = cv2.VideoCapture('768x576.avi')
    cap = cv2.VideoCapture(0)
    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret: #if vid finish repeat
            frame = cap = cv2.VideoCapture(0) #cv2.VideoCapture("768x576.avi")
            continue
        while True: 
            print(filename)
            ret, image = video.read()
            locations = face_recognition.face_locations(image, model=MODEL)
            encodings = face_recognition.face_encodings(image,)

            for face_encoding, face_location in zip(encodings, locations):

                # We use compare_faces (but might use face_distance as well)
                # Returns array of True/False values in order of passed known_faces
                results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                match = None
                if True in results:
                    match = known_names[results.index(True)]
                    print(f' - {match} from {results}')
                    print(f'Match Found : {match}')
                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])
                    color = name_to_color(match)
                    cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
                    
                    top_left = (face_location[3], face_location[2])
                    bottom_right = (face_location[1], face_location[2] + 22)
                    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                    cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
           break
   
@app.route('/video_ID')
def video_id():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_id(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#///////// ROUTE ERROR \\\\\\\\\\\\\\\\
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html")

@app.errorhandler(500)
def Internal_Server_Error(e):
    return render_template("500.html")

if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run(host='localhost', port=8080)