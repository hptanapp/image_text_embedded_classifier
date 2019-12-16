import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, array_to_img

# tf.enable_eager_execution()
indir = r"D:\Projects\image_text_model\data\inputs\images"

class custom_img_txt_generator:
    
    def __init__(self, img_root_dir, batch_size=64, img_size=(512, 512)):
        self.img_root_dir = img_root_dir
        self.img_size = img_size
        self.batch_size = batch_size

    def get_num_samples(self, list_data):
        return len(list_data)

    ## func to read image to array
    def img_loader(self, in_array):
        sub_img_path = os.path.join(in_array[0], in_array[1])
        img_path = os.path.join(self.img_root_dir, sub_img_path)
        if os.path.exists(img_path):
            img = load_img(img_path)
            img = tf.convert_to_tensor(np.asarray(img))
            image = tf.image.resize(img, self.img_size)
            return image
        else:
            return 0

    ## func to read text to array
    def txt_loader(self, in_array):
        txt = in_array[-1].split(";")
        return txt

    ## func to read output class to array
    def get_output(self, in_array):
        return in_array[0]

    ## data generator
    def image_text_generator(self, list_data):
        
        while True:
            ## Select files (paths/indices) for the batch
            print ("image_text_generator")
            batch_paths = np.random.choice(a = list_data, size = self.batch_size)

            batch_input = [] ## input format ([X_train_image, X_train_text])
            batch_output = [] ## output format (class)
            
            ## Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                input_path = input_path.split(",")
                ## get image data
                input_img = self.img_loader(input_path)

                ## get text data
                input_text = self.txt_loader(input_path)

                ## construct input
                input = np.array([input_img]), np.array([input_text])
                
                ## get output data
                output = self.get_output(input_path)
                
                batch_input += [ input ]
                batch_output += [ output ]
            
            ## Return a tuple of (input,output) to feed the network
            batch_x = batch_input
            batch_y = np.array( batch_output )

            yield batch_x, batch_y
            # return ( batch_x, batch_y )

if __name__ == "__main__":
    print ("started")
    infilepath = "D:/Projects/image_text_model/data/inputs/images/ocr.csv"
    with open(infilepath) as f:
        lines = f.read().splitlines()
    # print (lines)
    generator_class = custom_img_txt_generator(indir, batch_size=1)
    return_data = generator_class.image_text_generator(lines)
    data = return_data.__next__()
    print (data)
    print ("data[0]")
    print (data[0])
    print (np.shape(data[0]))
    print (np.shape(data[0][0]))
    # print (np.shape(data[0][1]))
    print ("data[1]")
    print (data[1])
    print (np.shape(data[1]))