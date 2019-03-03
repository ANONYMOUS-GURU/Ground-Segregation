import cv2
import numpy as np 

def labelize(img,size1,size2,scale_factor_distance=25,num_cluster=3,rand=6,batch_size=1000):
    a = cv2.medianBlur(img,3)
    a = cv2.medianBlur(a,3)
    a = cv2.medianBlur(a,3)
    
    loc1=np.zeros(shape=[size1,size2])
    loc2=np.zeros(shape=[size1,size2])
    
    for y in range(size1):
        loc1[:,y]=(y+1)/scale_factor_distance
    for y in range(size2):
        loc2[y,:]=(y+1)/scale_factor_distance
    
    a=np.reshape(a,[-1,1])
    loc1=np.reshape(loc1,[-1,1])
    loc2=np.reshape(loc2,[-1,1])
    a=np.concatenate([a,loc1,loc2],axis=1)
    
    multiplier=int(np.floor(255/num_cluster))
    
    from sklearn.cluster import MiniBatchKMeans
    kmb=MiniBatchKMeans(n_clusters=num_cluster,batch_size=batch_size,random_state=rand)
    
    label=kmb.fit_predict(a)*multiplier
    label=label.reshape([size1,size2])
    return label
