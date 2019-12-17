import os
import sys
import numpy as np
import cv2


indir = r"D:\Projects\image_text_model\data\inputs\images"
img_size = (512,512)
img_shape = (512,512,3)

## func to read image to array
def img_loader(in_img_path):
    if os.path.exists(in_img_path):
        image = cv2.imread(in_img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size) 
        if image.shape != img_shape:
            print (">>>>>>>>>HOLD UP !!!")
            print (in_img_path)
            print ("image shape : " + str(image.shape))
        return image
    else:
        print (">>>>>>>>>HOLD UP dafuq up !!!")
        print ("Failed")
        return 0

print ("==========Processing from directory==========")
for dirfile in os.listdir(indir):
    filename = os.fsdecode(dirfile)
    curclass = filename
    subdir = os.path.join(indir, filename)
    
    if os.path.isdir(subdir):
        for subfile in os.listdir(subdir):
            
            if subfile.endswith(".jpg") or subfile.endswith(".png") or subfile.endswith(".jpeg"):
                curfilename = subfile
                curimagefile = os.path.join(subdir, subfile)
                
                img_loader(curimagefile)
print ("Done !!")


def img_loader(in_array):
    sub_img_path = os.path.join(in_array[0], in_array[1])
    img_path = os.path.join(indir, sub_img_path)
    if os.path.exists(img_path):

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size) 
        return image
    else:
        return img_path

print ("==========Processing from csv==========")
csv_infilepath = "D:/Projects/image_text_model/data/inputs/images/ocr.csv"
with open(csv_infilepath) as f:
    lines = f.read().splitlines()
for input_path in lines:
    input_path = input_path.split(",")
    image = img_loader(input_path)
    try:
        if image.shape != img_shape:
            print (">>>>>>>>>HOLD UP !!!")
            print (in_img_path)
            print ("image shape : " + str(image.shape))
    except:
        print (">>>>>>>>>wait a minutes !! dafuq up !!!")
        print ("no shape ?")
        print (input_path)
        print (image)

print ("Done !!")
