import cv2
import numpy as np

def segregate(label,strip_size,size1,size2):
    from scipy import stats
    ch=np.array(label)   
    ch=ch.reshape([size1,size2])
    x=ch[size1-strip_size:]
    mode=stats.mode(x.reshape([-1]))[0][0]
    ch[ch!=mode]=-1
    ch[ch==mode]=255
    ch=np.clip(ch,0,255)
    return ch 
