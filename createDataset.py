import os
import glob
from PIL import Image
import h5py
import random
import copy
import numpy as np

n = 6000 ## number of images
ntrain = 5500
nval = 500
c = 3 ## channel
img_size = 225 ## image size

out_train="/home/insighturu/data/story_break_data/BBC_Planet_Earth_Dataset/train.h5"
out_val= '/home/insighturu/data/story_break_data/BBC_Planet_Earth_Dataset/val.h5'

## read all the image paths
imgs = glob.glob('/nfs_mount/datasets/other/medical/nlm/NLMCXR_png/*.png') ## List

ftrain = h5py.File(out_train, "w")
trainset = ftrain.create_dataset("images", (ntrain,c,img_size,img_size), dtype='f')

fval = h5py.File(out_val, "w")
valset = fval.create_dataset("images", (nval,c,img_size,img_size), dtype='f')

## shuffle the data
random.shuffle(imgs)

## get required number of images
imgs = imgs[:n]

train_imgs = copy.deepcopy(imgs[:ntrain])
print(train_imgs[0:10])
with open(out_train_list, 'w') as txt:
    txt.writelines('\n'.join([img for img in train_imgs]))

val_imgs = copy.deepcopy(imgs[ntrain:])
with open(out_val_list, 'w') as txt:
    txt.writelines('\n'.join([img for img in val_imgs]))

del imgs

print("creating training data")
for i, im_path in enumerate(train_imgs):
    ## resize
    ## add it to dataset
    im = Image.open(im_path)
    r,g,b = im.split()
    im = Image.merge("RGB", (b.resize((img_size,img_size)),g.resize((img_size,img_size)),r.resize((img_size,img_size))))
    out= np.asarray(im.getdata()).reshape(3,im.size[0], im.size[1])
    trainset[i] = out

    if i%200==0:
        print "%0.1f percent completed of %d"%(100*i/ntrain, ntrain)

print("creating validation data")
for i, im_path in enumerate(val_imgs):
    ## resize
    ## add it to dataset
    im = Image.open(im_path)
    r,g,b = im.split()
    im = Image.merge("RGB", (b.resize((img_size,img_size)),g.resize((img_size,img_size)),r.resize((img_size,img_size))))
    out= np.asarray(im.getdata()).reshape(3,im.size[0], im.size[1])
    valset[i] = out

    if i%200==0:
        print "%0.1f percent completed of %d"%(100*i/nval, nval)

## close the files
ftrain.close()
fval.close()
