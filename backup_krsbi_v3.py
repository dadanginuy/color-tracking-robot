#import paket yang diperlukan
from picamera.array import PiRGBArray     
from picamera import PiCamera
import RPi.GPIO as GPIO
import time
import cv2
import cv2.cv as cv
import numpy as np
import serial

GPIO.setmode(GPIO.BOARD)
GPIO.setup(12, GPIO.IN)

ser = serial.Serial(
              
               port='/dev/ttyS0',
               baudrate = 9600,
               parity=serial.PARITY_NONE,
               stopbits=serial.STOPBITS_ONE,
               bytesize=serial.EIGHTBITS,
               timeout=1
           )

#Image analysis work

#orange color
orange_lower=np.array([0,0,0],np.uint8)
orange_upper=np.array([10,255,255],np.uint8)
	
#white color
white_lower=np.array([0,0,115],np.uint8)
white_upper=np.array([255,255,255],np.uint8)

#cyan color
cyan_lower=np.array([0,0,115],np.uint8)
cyan_upper=np.array([255,255,255],np.uint8)

#magenta color
magenta_lower=np.array([0,0,115],np.uint8)
magenta_upper=np.array([255,255,255],np.uint8)

def maju():
    ser.write('maju\n')

def mundur():
    ser.write('mundur\n')  

def stop():
    ser.write('stop\n')

def kiri():
    ser.write('kiri\n')

def kanan():
    ser.write('kanan\n')

def putar_kanan():
    ser.write('putarkanan\n')

def putar_kiri():
    ser.write('putarkiri\n')

def tendang():
    ser.write('tendang\n')

def center():
    ser.write('center\n')
    
def segment_colour_orange(frame):    #returns only the red colors in the frame
    hsv_roi =  cv2.cvtColor(frame, cv2.cv.CV_BGR2HSV)
    mask_1 = cv2.inRange(hsv_roi, np.array([160, 160,10]), np.array([190,255,255]))
    ycr_roi=cv2.cvtColor(frame,cv2.cv.CV_BGR2YCrCb)
    mask_orange=cv2.inRange(ycr_roi, orange_lower, orange_upper)

    #morphological transfomation, dilation
    kern_dilate = np.ones((8,8),"uint8")

    orange=cv2.dilate(mask_orange,kern_dilate)
    res2=cv2.bitwise_and(frame,frame, mask=orange)

    mask = mask_orange | mask_1
    cv2.imshow('mask',mask)
    return mask

def segment_colour_white(frame):    #returns only the red colors in the frame
    hsv_roi =  cv2.cvtColor(frame, cv2.cv.CV_BGR2HSV)
    mask_1 = cv2.inRange(hsv_roi, np.array([160, 160,10]), np.array([190,255,255]))
    ycr_roi=cv2.cvtColor(frame,cv2.cv.CV_BGR2HSV)
    mask_white=cv2.inRange(ycr_roi, white_lower, white_upper)
	
    #morphological transfomation, dilation
    kern_dilate = np.ones((8,8),"uint8")
    kern_erode  = np.ones((3,3),"uint8")
	
    white=cv2.dilate(mask_white,kern_dilate)
    white=cv2.erode(mask_white,kern_erode)
    res1=cv2.bitwise_and(frame,frame, mask=white)

    mask = mask_white | mask_1
    cv2.imshow('mask',mask)
    return mask

def segment_colour_cyan(frame):    #returns only the red colors in the frame
    hsv_roi =  cv2.cvtColor(frame, cv2.cv.CV_BGR2HSV)
    mask_1 = cv2.inRange(hsv_roi, np.array([160, 160,10]), np.array([190,255,255]))
    ycr_roi=cv2.cvtColor(frame,cv2.cv.CV_BGR2HSV)
    mask_cyan=cv2.inRange(ycr_roi, cyan_lower, cyan_upper)
	
    #morphological transfomation, dilation
    kern_dilate = np.ones((8,8),"uint8")
    kern_erode  = np.ones((3,3),"uint8")
	
    cyan=cv2.dilate(mask_cyan,kern_dilate)
    cyan=cv2.erode(mask_cyan,kern_erode)
    res1=cv2.bitwise_and(frame,frame, mask=cyan)

    mask = mask_cyan | mask_1
    cv2.imshow('mask',mask)
    return mask
	
def segment_colour_magenta(frame):    #returns only the red colors in the frame
    hsv_roi =  cv2.cvtColor(frame, cv2.cv.CV_BGR2HSV)
    mask_1 = cv2.inRange(hsv_roi, np.array([160, 160,10]), np.array([190,255,255]))
    ycr_roi=cv2.cvtColor(frame,cv2.cv.CV_BGR2HSV)
    mask_magenta=cv2.inRange(ycr_roi, magenta_lower, magenta_upper)
	
    #morphological transfomation, dilation
    kern_dilate = np.ones((8,8),"uint8")
    kern_erode  = np.ones((3,3),"uint8")
	
    magenta=cv2.dilate(mask_magenta,kern_dilate)
    magenta=cv2.erode(mask_magenta,kern_erode)
    res1=cv2.bitwise_and(frame,frame, mask=magenta)

    mask = mask_magenta | mask_1
    cv2.imshow('mask',mask)
    return mask

def find_white(white): #returns the white colored circle
    largest_contour=0
    cont_index=0
    contours, hierarchy = cv2.findContours(white, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area >largest_contour) :
            largest_contour=area
           
            cont_index=idx
                              
    r=(0,0,2,2)
    if len(contours) > 0:
        r = cv2.boundingRect(contours[cont_index])
    return r,largest_contour

def find_orange(orange): #returns the orange colored circle
    largest_contour=0
    cont_index=0
    contours, hierarchy = cv2.findContours(orange, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area >largest_contour) :
            largest_contour=area
            cont_index=idx   
                              
    r=(0,0,2,2)
    if len(contours) > 0:
        r = cv2.boundingRect(contours[cont_index])
    return r,largest_contour

def find_cyan(cyan): #returns the cyan colored circle
    largest_contour=0
    cont_index=0
    contours, hierarchy = cv2.findContours(cyan, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area >largest_contour) :
            largest_contour=area
            cont_index=idx   
                              
    r=(0,0,2,2)
    if len(contours) > 0:
        r = cv2.boundingRect(contours[cont_index])
    return r,largest_contour
	
def find_magenta(magenta): #returns the magenta colored circle
    largest_contour=0
    cont_index=0
    contours, hierarchy = cv2.findContours(magenta, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area >largest_contour) :
            largest_contour=area
            cont_index=idx   
                              
    r=(0,0,2,2)
    if len(contours) > 0:
        r = cv2.boundingRect(contours[cont_index])
    return r,largest_contour
	
def target_hist(frame):
    hsv_img=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   
    hist=cv2.calcHist([hsv_img],[0],None,[50],[0,255])
    return hist

def scan_gawang():
    global centre_x1
    global centre_y1
    centre_x1=0.
    centre_y1=0.
    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    msk_white=segment_colour_white(frame)      #masking orange the frame
    loct1,area1=find_orange(msk_white)
    x1,y1,w1,h1=loct1

    simg3 = cv2.rectangle(frame, (x1,y1), (x1+w1,y1+h1), 255,2)
    centre_x1=x1+((w1)/2)
    centre_y1=y1+((h1)/2)
    cv2.circle(frame,(int(centre_x1),int(centre_y1)),3,(0,110,255),-1)
    cv2.putText(frame, "Gawang",(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    centre_x1-=80
    centre_y1-=80
    print centre_x1,centre_y1,w1*h1
	
def scan_cyan():
    global centre_x2
    global centre_y2
    centre_x2=0.
    centre_y2=0.
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    msk_cyan=segment_colour_cyan(frame)      #masking cyan the frame
    loct2,area2=find_cyan(msk_cyan)
    x2,y2,w2,h2=loct2

    simg3 = cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), 255,2)
    centre_x2=x2+((w2)/2)
    centre_y2=y2+((h2)/2)
    cv2.circle(frame,(int(centre_x2),int(centre_y2)),3,(0,110,255),-1)
    cv2.putText(frame, "cyan",(x2,y2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    centre_x2-=80
    centre_y2-=80
    print centre_x2,centre_y2,w2*h2

def scan_magenta():
    global centre_x3
    global centre_y3
    centre_x3=0.
    centre_y3=0.
    hsv3 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    msk_magenta=segment_colour_magenta(frame)      #masking magenta the frame
    loct3,area3=find_magenta(msk_magenta)
    x3,y3,w3,h3=loct3

    simg4 = cv2.rectangle(frame, (x1,y1), (x1+w1,y1+h1), 255,2)
    centre_x3=x3+((w3)/2)
    centre_y3=y3+((h3)/2)
    cv2.circle(frame,(int(centre_x3),int(centre_y3)),3,(0,110,255),-1)
    cv2.putText(frame, "cyan",(x3,y3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    centre_x3-=80
    centre_y3-=80
    print centre_x3,centre_y3,w3*h3
    
#CAMERA CAPTURE
#initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (160, 120)
camera.framerate = 40
rawCapture = PiRGBArray(camera, size=(160, 120))
 
# allow the camera to warmup
time.sleep(0.001)
maju() 
# capture frames from the camera
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
     #grab the raw NumPy array representing the image, then initialize the timestamp and occupied/unoccupied text
     frame = image.array
     global centre_x
     global centre_y
     centre_x=0.
     centre_y=0.
     hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     msk_orange=segment_colour_orange(frame)      #masking orange the frame
     loct,area=find_orange(msk_orange)
     x,y,w,h=loct
     
     
     if GPIO.input(12)==0: 
       			  
      		if(w*h)<90:	
			putar_kanan()
			found=0
	    
      		else:
            		found=1
            		simg2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
            		centre_x=x+((w)/2)
            		centre_y=y+((h)/2)
            		cv2.circle(frame,(int(centre_x),int(centre_y)),3,(0,110,255),-1)
	    		cv2.putText(frame, "Ball",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            		centre_x-=80
            		centre_y-=80
            		print centre_x,centre_y,w*h
	    		maju()
      		initial=400
      		flag=0
      
      		if(centre_x<=-10 or centre_x>=10):
              		if(centre_x<0):
                   		flag=0
                   		putar_kiri()
                   		#time.sleep(0.020)
              		elif(centre_x>0):
                   		flag=1
                   		putar_kanan()
                   		#time.sleep(0.020)      

      		if(found==0):
            		#if the ball is not found and the last time it sees ball in which direction, it will start to rotate in that direction
            		if flag==0:
                  		putar_kiri()
                  		time.sleep(0.05)
            		else:
                  		putar_kanan()
                  		time.sleep(0.05)
     if GPIO.input(12)==1:
			maju()
			global centre_x1
			global centre_y1
			centre_x1=0.
			centre_y1=0.
			hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			msk_white=segment_colour_white(frame)      #masking orange the frame
			loct1,area1=find_orange(msk_white)
			x1,y1,w1,h1=loct1
			initial=400
			flag=0
		
			if(w1*h1<40):
				found=0
				putar_kanan()
			else:
				maju()
				simg3 = cv2.rectangle(frame, (x1,y1), (x1+w1,y1+h1), 255,2)
    				centre_x1=x1+((w1)/2)
    				centre_y1=y1+((h1)/2)
    				cv2.circle(frame,(int(centre_x1),int(centre_y1)),3,(0,110,255),-1)
    				cv2.putText(frame, "Gawang",(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    				centre_x1-=80
    				centre_y1-=80
    				print centre_x1,centre_y1,w1*h1

			if(2000<(w1*h1)<10000):
				tendang()
		
                 
   		
     cv2.imshow("draw",frame)    
     rawCapture.truncate(0)  # clear the stream in preparation for the next frame
      
     if(cv2.waitKey(1) & 0xff == ord('q')):
            		break

GPIO.cleanup() #free all the GPIO pins