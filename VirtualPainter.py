import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
import cvzone as cvz


# import all our images and make a list.
folder = "images"
myList = os.listdir(folder)
overlayList = []
for imgPath in myList:
    image = cv2.imread(f"{folder}/{imgPath}")
    overlayList.append(image)

# original Images , where nothing is selected
header = overlayList[0]

# COLOR
drawColor = (255,0,255)
# brushThickness
brushThickness = 15
# eraserThickness
eraserThickness = 50
# xp,yp
xp ,yp = 0 , 0
# canvas
imgCanvas = np.zeros((720,1280,3),np.uint8) # 0 to 255


# Read the camera
cap  = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# hand detector  
detector = htm.handDetector(min_detection_confidence=0.85)
# Iterating through frames
while True:
    # 1. import Image
    success , img = cap.read()
    img = cv2.flip(img, 1) # right will left , so that it will be easy to draw.

    # 2. Find the landmarks.
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    
    if len(lmList)!=0:
        print(lmList)

        # tip of the index finger , middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]


        # 3. Checking which fingers are up , because we wagetitnt to draw when our index finger is up, and we want to select only when two fingers are up.
        fingers = detector.fingersup()
        print(fingers)


        # 4. If selection mode   - two fingers are up , then we have to select.
        if fingers[1] and fingers[2]:
            # 9. so that we can draw lines without remebering the previous pne, the problem was, the point where we left last, if we draw somewhere else it would draw  a line .
            xp , yp = 0,0
            # draw rectangle to know its selection mode.
            print("SELECTION MODE")
            # 6. if we are top of the image
            if y1 < 125: # we are in the header.
                if 0 < x1 < 180: # it is clicking first one.
                    header = overlayList[0]
                    drawColor = (0,0,255) # change color of the rectangle and circle according to your selection.
                elif 210 < x1 < 400:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 420 < x1 < 585:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 600 < x1 < 845:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)


        
        # 5. If drawing mode - when index finger is up
        if fingers[1] and fingers[2]==False:
            # draw circle to know its selection mode.
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("DRAWING MODE")
            
            # 7. drawing lines.
            if xp == 0 and yp == 0: # This will indicate that we have started drawing and the starting point is where our index finger is.
                xp,yp = x1,y1
            cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
            
            
            
            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)               
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)

            
            
            
            # 8. now we have to keep updating the previous point so that it can draw a line.
            xp , yp = x1, y1 # as we change our poistion the previous also changes.

           # """ when you run this, you will face a problem that it is drawing a line but it is disappearing the next frame , so need a extra layer to draw. """
        
        
        
        
        # 10. Clear Canvas when all fingers are up
        if all (x >= 1 for x in fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    
    
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
    
    
    # Set the header image
    img[0:125, 0:1280] = header
    cv2.imshow("image",img)
    cv2.imshow("canvas",imgCanvas)
    cv2.imshow("Inv", imgInv)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()