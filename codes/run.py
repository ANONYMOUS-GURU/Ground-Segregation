import time
import argparse
import cv2
import numpy as np

from segregation import segregate
from cluster import labelize 

parser=argparse.ArgumentParser(description='Ground Segregation')
parser.add_argument('-strip_size',type=int,required=False,default=100)
parser.add_argument('-size1',type=int,required=False,default=400)
parser.add_argument('-size2',type=int,required=False,default=400)
parser.add_argument('-k_value',type=int,required=False,default=3)
parser.add_argument('-cam',type=int,required=False,default=0)
args=parser.parse_args()


cam=args.cam
size1=args.size1
size2=args.size2
k_value=args.k_value
strip_size=args.strip_size

cap=cv2.VideoCapture(cam)
x=0
while True:
    _,frame=cap.read()
    a=time.time()
    img1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img1=cv2.resize(img1,(size1,size2), interpolation = cv2.INTER_AREA)
    label=labelize(img1,num_cluster=k_value,size1=size1,size2=size2)
    ground=segregate(label,strip_size=strip_size,size1=size1,size2=size2)
    
    cv2.imshow('original_feed',np.array(img1, dtype = np.uint8 ))
    #cv2.imshow('label',np.array(label, dtype = np.uint8 ))
    cv2.imshow('ground',np.array(ground, dtype = np.uint8 ))
    print('frame rate ' ,1/(time.time()-a))
    if cv2.waitKey(1)==27:
        break
    
    x+=1

cv2.destroyAllWindows()
cap.release()