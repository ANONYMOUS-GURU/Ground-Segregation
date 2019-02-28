
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
#%matplotlib inline
import time

a=time.time()
size1=400
size2=400
scale_factor_distance=25


for x in range(1,8):
    a=cv2.imread('a{}.jpg'.format(x))
    a=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)       ##### gray

    loc1=np.zeros(shape=[size1,size2])
    loc2=np.zeros(shape=[size1,size2])

    a=cv2.resize(a,(size1,size2), interpolation = cv2.INTER_AREA)
    #print(a.shape)
    a = cv2.medianBlur(a,3)
    a = cv2.medianBlur(a,3)
    a = cv2.medianBlur(a,3)
    for y in range(size1):
        loc1[:,y]=(y+1)/scale_factor_distance
    for y in range(size2):
        loc2[y,:]=(y+1)/scale_factor_distance
    
    a=np.reshape(a,[-1,1])
    loc1=np.reshape(loc1,[-1,1])
    loc2=np.reshape(loc2,[-1,1])
    #print(loc1.shape)
    #print(loc2.shape)
    #a=np.reshape(a,[size1*size2,3])    #####
    a=np.concatenate([a,loc1,loc2],axis=1)
    print(a.shape)
    if x==1:
        final=a
    else:
        final=np.hstack((final,a))

print(final.shape)

num_cluster=3
multiplier=int(np.floor(255/num_cluster))


from sklearn.cluster import MiniBatchKMeans
kmb=MiniBatchKMeans(n_clusters=num_cluster,batch_size=1000,random_state=6)

plt.figure(figsize=(100,100))
i=0

for x in range(1,4):
    a=time.time()
    label=kmb.fit_predict(final[:,x*3:(x+1)*3])*multiplier #### 5 to 3
    label=np.reshape(label,[size1,size2])
    img=np.reshape(final[:,x*3],[size1,size2])     #### 
    i+=1
    plt.subplot(8,2,i)
    plt.imshow(img)
    i+=1
    plt.subplot(8,2,i)
    plt.imshow(label)

plt.show()

