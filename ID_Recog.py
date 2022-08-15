#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from geopy.geocoders import Nominatim
from imutils.object_detection import non_max_suppression
import os
import cv2
import re
import subprocess
import time
import imutils
import pytesseract
import numpy as np
import threading
import scipy.ndimage as scp

langSel = 'eng'

idTaiwan = r"^[A-Z]{1}[0-9]{7}$"
idChina = r"^[A-Z]{2}[0-9]{6}$"
idMexico = r"^[A-Z]{2}[0-9]{5}$"
rgIDMatch = [idTaiwan]

winname = 'Press lowercase \'Q\' to manual exit.'
card_width = 580
card_height = 840
setWidth = 960
setHeight = 720
extend = 15

net = cv2.dnn.readNet(os.path.dirname(os.path.abspath(__file__)) + '\\'+  'frozen_east_text_detection.pb')
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

def checkWindowsGeolocate():
    accuracy = 1
    start = time.time()
    while True:
        pshellcomm = ['powershell']
        pshellcomm.append('add-type -assemblyname system.device; '\
                        '$loc = new-object system.device.location.geocoordinatewatcher; '\
                        '$loc.start(); '\
                        'while(($loc.status -ne "Ready") -and ($loc.permission -ne "Denied")) '\
                        '{start-sleep -milliseconds 10}; '\
                        '$acc = %d; '\
                        'while($loc.position.location.horizontalaccuracy -gt $acc) '\
                        '{start-sleep -milliseconds 10; $acc = [math]::Round($acc*1.5)}; '\
                        '$loc.position.location.latitude; '\
                        '$loc.position.location.longitude; '\
                        '$loc.position.location.horizontalaccuracy; '\
                        '$loc.stop()' % (accuracy) )

        p = subprocess.Popen(pshellcomm, stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text=True)
        out = p.communicate()[0]
        (lat, lng) = out.split('\n')[0:2]
        if lat:
            break
        else:
            if time.time() - start > 10:
                break
        
    return lat, lng


def getArea():
    getPlace = ''
    latitude, longitude = checkWindowsGeolocate()
    try:
        locator = Nominatim(user_agent='user')
        location = locator.reverse(str(latitude)+', '+str(longitude), language='en')
        addr = location.address.split(',')
        addr = [s.strip() for s in addr]
        addr = list(reversed(addr))
        getPlace = ", ".join(addr[2:4])
    except (Exception, BaseException) as e:
        pass
    finally:
        locator.adapter.session.close()
    
    return getPlace + '/'


def EAST_detector(image, resizeLength=320, min_confidence=0.5):
    def bounding(scores, geometry):
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(0, numRows):

            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                if scoresData[x] < min_confidence:
                    continue
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        
        return rects, confidences
    
    (h, W) = image.shape[:2]
    rW = W / float(resizeLength)
    rH = h / float(resizeLength)

    image = cv2.resize(image, (resizeLength, resizeLength))
    (h, W) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    scores, geometry = net.forward(layerNames)
    rects, confidences = bounding(scores, geometry)
    
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    subRects = []
    for rec in boxes:
        sx = int(rec[0] * rW)
        sy = int(rec[1] * rH)
        ex = int(rec[2] * rW)
        ey = int(rec[3] * rH)

        if (ey - sy) > (ex - sx):
            continue
        
        subRects.append((sx, sy, ex, ey))

    if subRects:
        subRects = sorted(subRects, key=lambda x: x[1], reverse=True)
    
    return subRects


def getFrame(success):
    global frame, exitFlag
    while success and (cv2.waitKey(1) != ord('q')) and exitFlag is False:
        success, frame = vdCap.read()
        cropLine = cv2.rectangle(frame.copy(), (start_x, start_y), (end_x, end_y), crop_color, thickness=2)
        if subBoxes:
            for (sx, sy, ex, ey) in subBoxes:
                cropLine = cv2.rectangle(cropLine, (start_x + sx, start_y + sy), (start_x + ex, start_y + ey), color=(50, 255, 70), thickness=2)
        
        cropLine = cv2.flip(cropLine, 1)
        cv2.putText(cropLine, status, (start_x-20, start_y-3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale=0.7, thickness=2)
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 50, 30)
        cv2.imshow(winname, cropLine)
    
    exitFlag = True
    vdCap.release()
    cv2.destroyAllWindows()
    time.sleep(0.01)


def filterIDToken(strList):
    for strTK in strList:
        tempStr = ''.join(filter(str.isalnum, strTK))
        if not tempStr:
            continue
        if re.match('|'.join(rgIDMatch), tempStr):
            return strTK
    return ''


def getGreyScale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0, sigmaY=0)


def thresholding(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=4)


# fatter black line
def erode(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# fatter white area
def dilate(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def correct_skew(image, delta=1, limit=45):
    def determine_score(arr, angle):
        data = scp.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        score = determine_score(thresh, angle)[1]
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    rotated = scp.rotate(image, best_angle, reshape=False, order=0)

    return rotated


def Recog(img):
    global status
    status = 'Recognizing ...'
    try:
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = imutils.resize(img, width=480)
        for step in range(0, 2):
            text = pytesseract.image_to_string(img, lang=langSel)
            rstID = filterIDToken(text.split())
            if rstID:
                return rstID
            
            if step == 0:
                img = correct_skew(img)
            elif step == 1:
                img = getGreyScale(img)
            elif step == 2:
                img = thresholding(img)
            elif step == 3:
                img = remove_noise(img)
            elif step == 4:
                img = erode(img)
            elif step == 5:
                img = dilate(img)
    except:
        status = ''
        pass
    return ''


if __name__ == "__main__":  
    # vdCap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    vdCap = cv2.VideoCapture(0)
    vdCap.set(cv2.CAP_PROP_FRAME_WIDTH, setWidth)
    vdCap.set(cv2.CAP_PROP_FRAME_HEIGHT, setHeight)

    if not vdCap.isOpened():
        exit(1)

    vdCapWidth = vdCap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vdCapHeight = vdCap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    crop_scale = (card_height / vdCapHeight) + 0.2

    success, frame = vdCap.read()
    if not success:
        exit(1)
    
    status = ''
    exitFlag = False
    gResult = ''
    subBoxes = [] 

    start_x = int(vdCapWidth / 2 - (card_width / crop_scale) / 2)
    start_y = int(vdCapHeight / 2 - (card_height / crop_scale) / 2)
    crop_color = (255, 255, 255)
    end_x = start_x + int(card_width / crop_scale)
    end_y = start_y + int(card_height / crop_scale)

    gfThread = threading.Thread(target=getFrame, args=(success,))
    gfThread.start()
    
    header = getArea()
    while exitFlag is False:
        status = ''
        subBoxes.clear()
        gResult = Recog(frame[start_y : end_y, start_x : end_x])
        if gResult:
            status = gResult
            crop_color = (38, 255, 130)
            exitFlag = True
            break
        else:
            targetFrame = frame[start_y : end_y, start_x : end_x]
            subBoxes = EAST_detector(targetFrame)
            for (sx, sy, ex, ey) in subBoxes.copy():
                gResult = Recog(targetFrame[sy : ey, (sx -  extend) : (ex + extend)])
                if gResult:
                    status = gResult
                    crop_color = (38, 255, 130)
                    exitFlag = True
                    break
                else:
                    subBoxes.clear()

    print(header + gResult)
  