# Include libraries we are going to use
import cv2
import numpy as np
from math import *
from classes.HoughBundler import *
class Characters_Detection:

    def __init__(self, inputImage, R2L):
        path = str(inputImage )
        self.img = cv2.imread(path)
        if(R2L):    
            self.img = cv2.flip(self.img, 1)
    	

    def getCharacters(self,OutputPath):
        img=self.img

        # .Gray effect and gaussian blur to smooth gaussian noise
        # .cv2.GaussianBlur(imageVariable,(width_of_the_kernel ,height_of_the_kernel),sigma)
        # .Increase sigma value to increase blurness

        # .height and width should be odd and can have different values. If ksize is set to [0 0], then ksize is computed from sigma values
        # .sigmaX    Kernel standard deviation along X-axis (horizontal direction).
        # .sigmaY    Kernel standard deviation along Y-axis (vertical direction). If sigmaY=0, then sigmaX value is taken for sigmaY
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)


        # KERNEL is a matrix you need to define the size of and the value of
        # dilations ->increase thickness
        # reoded ->decrease thickness
        #define kernel 
        kernel = np.ones((3,3),np.uint8)
        imgCanny2 = cv2.Canny(imgBlur,100,100)
        imgDialationX = cv2.dilate(imgCanny2,kernel,iterations=1)


        # threshold, which means the minimum vote it should get to be considered as a line.
        # return -> (x1,y1,x2,y2)
        # maxLineGap -> Maximum allowed gap between points on the same line to link them.
        minLineLength = (np.minimum(img.shape[0],img.shape[1]))/2
        maxLineGap = 5
        lines = cv2.HoughLinesP(imgDialationX,1,np.pi/180,300, 100000,minLineLength,maxLineGap)
        #print(lines)

        try:
        	a=HoughBundler()
        	if (len(lines) != 0 ):
        		FinalLines=a.completeLines(lines,imgDialationX)
        		print(len(FinalLines))
        		for line in lines:
        			x1,y1,x2,y2 =line[0]
        			cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        			imgContour1 = img.copy()
        			self.getContours(imgDialationX,FinalLines)
        except:
        	imgContour1 = img.copy()
        	self.getContoursNoL(imgDialationX)


        # RETR_EXTERNAL detect outer contours or outer details
        # CHAIN_APPROX_NONE dont make any approximation to the contours
        # -1 to draw all the contours 
        # 3 thickness
        # True -> all our shapes to be closed

    def getContours(self,img,FinalLines):
            contourArr=[]
            imgContour1 = img.copy()
            a=HoughBundler()
            contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                #print(area)
                if area>100:

                    peri = cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                    objCor = len(approx)
                    x,y,w,h = cv2.boundingRect(approx)
                    cCoordMidX=(x-(0.5*w))
                    cCoordMidY=(y-(0.5*h))
                    cDist=np.sqrt(np.square(cCoordMidX)+np.square(cCoordMidY))                    
                    contourArr.append([cDist,x,y,w,h,cCoordMidX,cCoordMidY])

            contourArr2=np.array(contourArr)
            sortedArr = contourArr2[contourArr2[:,0].argsort()]                
            orderAlph=0
            for sCnt in sortedArr:
                orderAlph=orderAlph+1
                cDist,x,y,w,h,cCoordMidX,cCoordMidY =sCnt
                #print("x=",x,"y=",y)
                x=np.int64(x)
                y=np.int64(y)
                w=np.int64(w)
                h=np.int64(h)
                cv2.rectangle(imgContour1,(x,y),(x+w,y+h),(0,255,0),2)
                cropped_Contour = self.img[y: y + h, x: x + w]
                enumu=1
        
                if(a.chk_I_V2(FinalLines)):
               	 for line in FinalLines:
                    		x1,y1,x2,y2 =line
                    		if(cCoordMidX>x1):enumu=enumu+1
                elif(a.chk_I_V2(FinalLines)==False):
                        for line in FinalLines:
                        	x1,y1,x2,y2 =line
                        	if(cCoordMidY>y1):enumu=enumu+1
                    
                image_Name = "L" + str(enumu) +"Q"+str(orderAlph)+ ".jpg"
                cv2.imwrite(image_Name, cropped_Contour) 


            return contourArr



    '''def cropContours(self,contourArr,FinalLines):
            imgContour1 = img.copy()
            contourArr2=np.array(contourArr)
            sortedArr = contourArr2[contourArr2[:,0].argsort()]                
            orderAlph=0
            for sCnt in sortedArr:
                orderAlph=orderAlph+1
                cDist,x,y,w,h,cCoordMidX,cCoordMidY =sCnt
                #print("x=",x,"y=",y)
                x=np.int64(x)
                y=np.int64(y)
                w=np.int64(w)
                h=np.int64(h)
                cv2.rectangle(imgContour1,(x,y),(x+w,y+h),(0,255,0),2)
                cropped_Contour = self.img[y: y + h, x: x + w]
                enumu=1
        
                if(a.chk_I_V2(FinalLines)):
               	 for line in FinalLines:
                    		x1,y1,x2,y2 =line
                    		if(cCoordMidX>x1):enumu=enumu+1
                elif(a.chk_I_V2(FinalLines)==False):
                        for line in FinalLines:
                        	x1,y1,x2,y2 =line
                        	if(cCoordMidY>y1):enumu=enumu+1
                    
                image_Name = "L" + str(enumu) +"Q"+str(orderAlph)+ ".jpg"
                cv2.imwrite(image_Name, cropped_Contour)'''




    def getContoursNoL(self,img):
        contourArr=[]
        a=HoughBundler()
        imgContour1 = img.copy()
        contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area>100:
                peri = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                objCor = len(approx)
                x,y,w,h = cv2.boundingRect(approx)
                cCoordMidX=(x-(0.5*w))
                cCoordMidY=(y-(0.5*h))
                cDist=np.sqrt(np.square(cCoordMidX)+np.square(cCoordMidY))                    
                contourArr.append([cDist,x,y,w,h,cCoordMidX,cCoordMidY])

        contourArr2=np.array(contourArr)
        sortedArr = contourArr2[contourArr2[:,0].argsort()]                
        orderAlph=0
        for sCnt in sortedArr:
                orderAlph=orderAlph+1
                cDist,x,y,w,h,cCoordMidX,cCoordMidY =sCnt
                x=np.int64(x)
                y=np.int64(y)
                w=np.int64(w)
                h=np.int64(h)
                cv2.rectangle(imgContour1,(x,y),(x+w,y+h),(0,255,0),2)
                cropped_Contour = self.img[y: y + h, x: x + w]
                image_Name = "L1Q"+str(orderAlph)+ ".jpg"
                cv2.imwrite(image_Name, cropped_Contour) 


