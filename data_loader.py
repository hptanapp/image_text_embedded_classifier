import os
import sys
import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder

# tf.enable_eager_execution()
indir = r"D:\Projects\image_text_model\data\inputs\images"

class custom_img_txt_generator:
    
    def __init__(self, img_root_dir, batch_size=64, img_size=(512, 512), txt_size=10000):
        self.img_root_dir = img_root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.txt_size = txt_size

    def get_num_samples(self, list_data):
        return len(list_data)

    ## func to read image to array
    def img_loader(self, in_array):
        sub_img_path = os.path.join(in_array[0], in_array[1])
        img_path = os.path.join(self.img_root_dir, sub_img_path)
        if os.path.exists(img_path):

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size) 
            return image

    ## func to read text to array
    def txt_loader(self, in_array):
        txt = in_array[-1].split(";")
        # txt_np = np.zeros(self.txt_size, dtype=object)
        # txt_np[:len(txt)] = txt
        return txt

    ## fill text dimension
    def txt_fill(self, in_text):
        txt_np = np.empty(self.txt_size, dtype=int)
        in_text = np.array(in_text).reshape(-1)
        txt_np[:len(in_text)] = in_text
        return txt_np

    ## func to tokenizer
    def get_tokenizer(self, list_data):
        texts = []
        for input_path in list_data:
            texts.append(input_path.split(",")[-1])
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        return tokenizer

    ## func to read output class to array
    def get_output(self, in_array):
        return in_array[0]

    ## func to get total prediction classes
    def get_total_classes(self, list_data):
        output_classes = []
        for input_path in list_data:
            output_classes.append(input_path.split(",")[0])
        output_classes = list(set(output_classes))
        output_classes.sort()
        output_classes = np.array(output_classes)
        output_classes = output_classes.reshape(-1, 1)
        return output_classes

    ## func to get onehotencode
    def get_onehot(self, output_classes):
        enc = OneHotEncoder()
        enc.fit(output_classes)
        return enc

    ## func to convert to onehot
    def get_onehot_array(self, onehot_enc, output):
        output = np.array(output)
        output = output.reshape(-1,1)
        output = onehot_enc.transform(output).toarray()
        output = output.reshape(-1, output.shape[-1])
        return output

    ## data generator
    def image_text_generator(self, list_data):
        
        while True:
            ## Select files (paths/indices) for the batch
            batch_paths = np.random.choice(a = list_data, size = self.batch_size)

            batch_input = [] ## input format ([X_train_image, X_train_text])
            batch_input_1 = []
            batch_input_2 = []
            batch_output = [] ## output format (class)

            ## get total prediction classes
            total_classes = self.get_total_classes(list_data)
            onehot_enc = self.get_onehot(total_classes)

            ## get tokenizer
            tokenizer = self.get_tokenizer(list_data)
            
            ## Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                input_path = input_path.split(",")
                ## get image data
                input_img = self.img_loader(input_path)

                ## get text data
                input_text = self.txt_loader(input_path)
                input_text = tokenizer.texts_to_sequences(input_text)
                input_text = self.txt_fill(input_text)

                ## construct input
                input = [input_img, np.array([input_text])]
                
                ## get output data
                output = self.get_output(input_path)
                
                batch_input += [ input ]
                batch_input_1 += [ input_img ]
                batch_input_2 += [ input_text ]
                batch_output += [ output ]
                
            batch_output = self.get_onehot_array(onehot_enc, batch_output)
            
            ## Return a tuple of (input,output) to feed the network
            batch_x = np.array( batch_input )
            batch_x_1 = np.array( batch_input_1 )
            batch_x_2 = np.array( batch_input_2 )
            batch_y = np.array( batch_output )

            # yield batch_x, batch_y
            yield [batch_x_1,batch_x_2], batch_y
            # return batch_x, batch_y

if __name__ == "__main__":
    print ("started")
    infilepath = "D:/Projects/image_text_model/data/inputs/images/ocr.csv"
    with open(infilepath) as f:
        lines = f.read().splitlines()
    # print (lines)
    generator_class = custom_img_txt_generator(indir, batch_size=10)
    return_data = generator_class.image_text_generator(lines)
    for i in range(10):
        print ("============================")
        print (i)
        print ("============================")

        data = return_data.__next__()
        # print (data)
        print ("data[0]")
        # print (np.shape(data))
        # print (np.shape(data[0]))
        # print (np.shape(data[0][0]))
        # print (np.shape(data[0][1]))
        print (data[0][0].shape)
        print (data[0][1].shape)
        print ("data[1]")
        # print (data[1])
        print (data[1].shape)