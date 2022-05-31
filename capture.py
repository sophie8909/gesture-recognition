#------------------------------------------------------------
# SEGMENT, RECOGNIZE and COUNT fingers from a video sequence
#------------------------------------------------------------

# organize imports
import cv2
from cv2 import blur
import imutils
import numpy as np
from sklearn.metrics import pairwise
import os
import random
import math
# global variables
bg = None

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    contours, hierarchy  = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(contours) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(contours, key=cv2.contourArea)
        return (thresholded, segmented)

#--------------------------------------------------------------
# To count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count             

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False
    queue_size = 50
    hand_queue = [0] * queue_size
    mode = 0

    # custom variable for sticker
    isStickerSet = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)


        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                
                # count the number of fingers
                fingers = count(thresholded, segmented)
                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                

                hand_queue[num_frames%queue_size] = fingers


        if hand_queue.count(1) >= queue_size//2:
            mode = 1
        elif hand_queue.count(2) >= queue_size//2:
            mode = 2
        elif hand_queue.count(3) >= queue_size//2:
            mode = 3
        elif hand_queue.count(4) >= queue_size//2:
            mode = 4
        elif hand_queue.count(5) >= queue_size//2:
            mode = 5
        # print(mode)
        result = clone.copy()
        # check for sticker status
        if mode != 2:
            isStickerSet = False

        if mode == 1:
            # 灰階
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        elif mode == 2:
            # 加貼圖
            sticker_folder = os.path.join(os.path.dirname(__file__), 'sticker')
            if isStickerSet == False:
                sticker_filename = random.choice(os.listdir(sticker_folder))
                isStickerSet = True

            sticker_path = os.path.join(sticker_folder, sticker_filename)
            sticker = cv2.imread(sticker_path)
            sticker = cv2.resize(sticker, (150, 150),
                                 interpolation=cv2.INTER_AREA)
            # Put the sticker at the right bottom corner
            x_offset = result.shape[1] - 170
            y_offset = result.shape[0] - 170
            x_end = x_offset + sticker.shape[1]
            y_end = y_offset + sticker.shape[0]
            result[y_offset:y_end, x_offset:x_end] = sticker
        elif mode == 3:
            brightness = 0
            contrast = 100
            
            B = brightness / 255.0
            C = contrast / 255.0
            K = math.tan((45 + 44 * C) / 180 * math.pi)
            
            result = (result - 127.5 * (1 - B)) * K + 127.5 * (1 + B)
            result = np.clip(result, 0, 255).astype(np.uint8)
             
        elif mode == 4:
            
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            inverted = 255 - gray
            blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
            inverted_blurred = 255 - blurred
            result = cv2.divide(gray, inverted_blurred, scale = 256.0)
            
            
        elif mode == 5:
            result = 255 - result
            
        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        cv2.imshow("Result", result)
        


        
        
        # increment the number of frames
        num_frames += 1

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()