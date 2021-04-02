import numpy as np
import cv2
import imutils
import os
import time
import pandas as pd
import datetime

current_car_amount = 0 # todo this is the added object to count the num of car
class TrafficCounter(object):
    current_car_amount = 0
    def __init__(self,video_source=('traffic.mp4'),
                 line_direction='H',
                 line_position=0.5, # todo line_position=0.5
                 video_width = 640,
                 min_area = 250, # todo min_area = 250
                 video_out='',
                 numCnts=10,
                 out_video_params={},
                 starting_frame=10,):
        self.crop_rect         = []       #stores the click coordinates where to crop the frame
        self.mask_points       = []       #stores the click coordinates of the mask to apply to cropped frame
        self.font              = cv2.FONT_HERSHEY_SIMPLEX
        self.numCnts = numCnts
        self.minArea           = min_area        

        self.video_source      = cv2.VideoCapture(video_source)

        self.frame = 0
        self.screenshot_folder = '_screenshots'
        self.video_out_folder  = '_videos'
        self.current_car_amount = 0 # todo to print the current car in current frame

        self._vid_width        = video_width       
        self._vid_height       = None        #PLACEHOLDER
        self.black_mask        = None        #PLACEHOLDER, user creates it by clicking on several points
        
        self.prev_centroids    = []          #this will contain the coordinates of the centers in the previos

        self.out_video_params  = out_video_params
        if len(video_out) < 1:
            self.video_out = False 
        else:
            self.video_out = True 
            self._out_vid_base_name = video_out
            self._set_video_writers()


    def _set_video_writers(self):
        fps           = self.video_source.get(cv2.CAP_PROP_FPS)
        video_ext     = self.out_video_params.get('extension','avi')
        string_fourcc = self.out_video_params.get('codec','mjpg')
        fourcc        = cv2.VideoWriter_fourcc(*string_fourcc)
        video_res     = (self._vid_width,self._vid_height)
        collage_res   = (self.collage_width,self.collage_height)


        
        self.out_bg_subtracted  = cv2.VideoWriter(os.path.join(self.video_out_folder,self._out_vid_base_name + '_bg_subtracted'  + '.' + video_ext),
                                                  fourcc,fps,video_res)
        self.out_threshold      = cv2.VideoWriter(os.path.join(self.video_out_folder,self._out_vid_base_name + '_threshold'      + '.' + video_ext),
                                                  fourcc,fps,video_res)
        self.out_bg_average     = cv2.VideoWriter(os.path.join(self.video_out_folder,self._out_vid_base_name + '_bg_average'     + '.' + video_ext),
                                                  fourcc,fps,video_res)
        self.out_bounding_boxes = cv2.VideoWriter(os.path.join(self.video_out_folder,self._out_vid_base_name + '_bounding_boxes' + '.' + video_ext),
                                                  fourcc,fps,video_res)
        self.out_collage        = cv2.VideoWriter(os.path.join(self.video_out_folder,self._out_vid_base_name + '_collage'        + '.' + video_ext),
                                                  fourcc,fps,collage_res)

    def _release_video_writers(self):
        self.out_bg_subtracted.release()
        self.out_threshold.release()
        self.out_bg_average.release()
        self.out_bounding_boxes.release()
        self.out_collage.release()




    def _compute_frame_dimensions(self):
        grabbed,img = self.video_source.read()
        while not grabbed:
            grabbed,img = self.video_source.read()
        img = imutils.resize(img,width=self._vid_width)
        self._vid_height = img.shape[0]
        self._vid_width  = img.shape[1]

    def _click_crop_event(self,event,x,y,flags,userdata):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.crop_rect.append((y,x))                  #Numpy manages the coordinates as (y,x) instead of (x,y)

    def _click_mask_event(self,event,x,y,flags,userdata):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.mask_points.append((x,y))
    
    def _draw_bounding_boxes(self,frame,contour_id,bounding_points,cx,cy,prev_cx,prev_cy):
        cv2.drawContours(frame,[bounding_points],0,(0,255,0),1)
        cv2.line(frame,(prev_cx,prev_cy),(cx,cy),(0,0,255),1)          #line between last position and current position
        cv2.circle(frame,(cx,cy),3,(0,0,255),4)
        cv2.putText(frame,str(contour_id),(cx,cy-15),self.font,0.4,(255,0,0),2)



    def bind_objects(self,frame,thresh_img):

        cnts,_ = cv2.findContours(thresh_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts,key = cv2.contourArea,reverse=True)[:self.numCnts]

        cnt_id         = 1
        cur_centroids  = []
        self.current_car_amount = 0 # todo to set
        for c in cnts:
            if cv2.contourArea(c) < self.minArea:           #ignore contours that are smaller than this area
                continue

            self.current_car_amount += 1 # todo to detect the current car amount
            #todo maybe the right print("current_car_amount : " + str(self.current_car_amount))
            rect   = cv2.minAreaRect(c)
            points = cv2.boxPoints(rect)                # This is the way to do it in opencv 3.1
            points = np.int0(points)

            #Getting the center coordinates of the contour box
            cx = int(rect[0][0])
            cy = int(rect[0][1])

            w,h = rect[1]                   #Unpacks the width and height of the frame

            C = np.array((cx,cy))
            cur_centroids.append((cx,cy))

            #Finding the centroid of c in the previous frame
            if len(self.prev_centroids)==0: 
                prev_cx,prev_cy = cx,cy
            elif len(cnts)==0: 
                prev_cx,prev_cy = cx,cy
            else:
                minPoint = None
                minDist = None
                for i in range(len(self.prev_centroids)):
                    dist = np.linalg.norm(C - self.prev_centroids[i])                #numpy's way to find the euclidean distance between two points
                    if (minDist is None) or (dist < minDist):
                        minDist = dist
                        minPoint = self.prev_centroids[i]
                #This if is meant to reduce overcounting errors
                if minDist < w/2:
                    prev_cx,prev_cy = minPoint
                else: 
                    prev_cx,prev_cy = cx,cy
                #prev_cx,prev_cy = minPoint



            cnt_id += 1
        #str(self.frame)[:str(self.frame).find('.') + 4]
        print("time " +  str(self.frame)[:str(self.frame).find('.') + 4] + " current_car_amount : " + str(self.current_car_amount))
        self.frame += 0.03
        self.prev_centroids = cur_centroids       #updating centroids for next frame

    def _set_up_masks(self):
        grabbed,img = self.video_source.read()
        while not grabbed:
            grabbed,img = self.video_source.read()

        img = imutils.resize(img,width=self._vid_width)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self._vid_height = img.shape[0]

        ##-------Show setup window
        k = None
        cv2.namedWindow('setup2',1)
        cv2.imshow('setup2',img)
        while k != ord('q') and k != ord('Q') and k != 27 and k != ('\n'):
            cv2.setMouseCallback('setup2',self._click_mask_event)
            k = cv2.waitKey(0) & 0xFF

        cv2.destroyWindow('setup2')

        roi_points = np.array([self.mask_points])
        self.black_mask = None
        if len(self.mask_points)!= 0:
            self.black_mask = np.zeros(img.shape,dtype=np.uint8)
            cv2.fillPoly(self.black_mask,roi_points,(255,255,255))

            self.raw_avg = np.float32(self.black_mask)
        else: self.raw_avg = np.float32(img)

        self.raw_avg = cv2.resize(self.raw_avg, (self._vid_width,self._vid_height))

    def make_collage_of_four(self,up_left_img,up_right_img,down_left_img,down_right_img):
        middle_width  = self._vid_width
        middle_height = self._vid_height
        total_height  = self.collage_frame.shape[0]
        total_width   = self.collage_frame.shape[1]

        if len(up_left_img.shape) < 3:
            up_l = cv2.cvtColor(up_left_img,cv2.COLOR_GRAY2BGR)
        else:
            up_l = up_left_img
        
        if len(up_right_img.shape) < 3:
            up_r = cv2.cvtColor(up_right_img,cv2.COLOR_GRAY2BGR)
        else:
            up_r = up_right_img

        if len(down_left_img.shape) < 3:
            down_l = cv2.cvtColor(down_left_img,cv2.COLOR_GRAY2BGR)
        else:
            down_l = down_left_img

        if len(down_right_img.shape) < 3:
            down_r = cv2.cvtColor(down_right_img,cv2.COLOR_GRAY2BGR)
        else:
            down_r = down_right_img

        self.collage_frame[0:middle_height,0:middle_width]                      = up_l    #setting up_left image
        self.collage_frame[0:middle_height,middle_width:total_width]            = up_r 
        self.collage_frame[middle_height:total_height,0:middle_width]           = down_l
        self.collage_frame[middle_height:total_height,middle_width:total_width] = down_r

    def main_loop(self):
        self._set_up_masks()
        rate_of_influence = 0.1## todo rate_of_influence = 0.01
        FRAME_CROPPED = False

        while True:

            grabbed,img = self.video_source.read()
            if not grabbed:
                break
            #--------------
            frame_id = int(self.video_source.get(1))        #get current frame index
            img = cv2.resize(img,(self._vid_width,self._vid_height))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            if FRAME_CROPPED:
                working_img = img[x:x+w,y:y+h]
            else:
                working_img = img.copy()
            if self.black_mask is not None:
                working_img = cv2.bitwise_and(working_img,self.black_mask)


            
            cv2.accumulateWeighted(working_img,self.raw_avg,rate_of_influence)
            background_avg = cv2.convertScaleAbs(self.raw_avg)           #reference background average image
            subtracted_img = cv2.absdiff(background_avg,working_img)

            ##-------Adding extra blur------
            subtracted_img = cv2.GaussianBlur(subtracted_img,(21,21),0)
            subtracted_img = cv2.GaussianBlur(subtracted_img,(21,21),0)
            subtracted_img = cv2.GaussianBlur(subtracted_img,(21,21),0)
            subtracted_img = cv2.GaussianBlur(subtracted_img,(21,21),0)

            ##-------Applying threshold
            _,threshold_img  = cv2.threshold(subtracted_img,30,255,0)

            #todo try no noise reduction
            ##-------Noise Reduction
            dilated_img      = cv2.dilate(threshold_img,None)
            dilated_img      = cv2.dilate(dilated_img,None)
            dilated_img      = cv2.dilate(dilated_img,None)
            dilated_img      = cv2.dilate(dilated_img,None)
            dilated_img      = cv2.dilate(dilated_img,None)

            ##-------Drawing bounding boxes and counting
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)          #Giving frame 3 channels for color (for drawing colored boxes)

            self.current_car_amount = 0
            self.bind_objects(img,dilated_img)

            ##-------Termination Conditions
            k = cv2.waitKey(25) & 0xFF
            if k == 27 or k == ord('q') or k == ord('Q'):
                break
            elif k == ord('s') or k == ord('S'):                #if the letter s/S is pressed, a screenshot of the current frame on each window will be saved to the current folder
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_masked_frame.jpeg"),working_img)
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_background_subtracted.jpeg"),subtracted_img)
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_threshold_applied.jpeg"),dilated_img)
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_background_average.jpeg"),background_avg)
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_car_counting.jpeg"),img)
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_collage.jpeg"),self.collage_frame)
            if k == ord(' '):   #if spacebar is pressed
                paused_key = cv2.waitKey(0) & 0xFF       #program is paused for a while
                if paused_key == ord(' '):    #pressing space again unpauses the program
                    pass

            if self.video_out:
                self.out_bg_subtracted.write(subtracted_img)
                self.out_threshold.write(dilated_img)
                self.out_bg_average.write(background_avg)
                self.out_bounding_boxes.write(img)
                self.out_collage.write(self.collage_frame)

        self.video_source.release()
        if self.video_out:
            self._release_video_writers()
        cv2.destroyAllWindows()