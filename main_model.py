import numpy as np
from keras.layers import Dropout
from keras import applications
from keras.layers import Dense, GlobalAveragePooling2D, merge, Input
from keras.layers import concatenate
from keras.models import Model

from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D
from tensorflow.keras import models
from keras.callbacks import ModelCheckpoint
# from tensorflow.keras import layers
import tensorflow
from keras_efficientnets import EfficientNetB0 # pip install keras-efficientnets
from keras_efficientnets import preprocess_input 
## custo,
import data_loader



max_words = 10000
epochs = 50
batch_size = 4

height = 512
width = 512
input_shape = (height, width, 3)

## define
img_root_dir = r"D:\Projects\image_text_model\data\inputs\images"
csv_infilepath = "D:/Projects/image_text_model/data/inputs/images/ocr.csv"
model_path = r"D:/Projects/image_text_model/models"

#### dataloader
with open(csv_infilepath) as f:
    lines = f.read().splitlines()
lines.pop(0)
custom_img_txt_generator_class_train = data_loader.custom_img_txt_generator(img_root_dir)
custom_img_txt_generator_class_test = data_loader.custom_img_txt_generator(img_root_dir)
num_samples = custom_img_txt_generator_class_train.get_num_samples(lines)
print ("total samples : " + str(num_samples))

num_classes = len(custom_img_txt_generator_class_train.get_total_classes(lines))

# X_train_image = ...  #images training input
# X_train_text = ... #text training input
# y_train = ... #training output
# num_classes = np.max(y_train) + 1



#### model

# Text input branch - just a simple MLP
text_inputs = Input(shape=(max_words,))
branch_1 = Dense(512, activation='relu')(text_inputs)

# Image input branch - a pre-trained Inception module followed by an added fully connected layer
# base_model = applications.InceptionV3(weights='imagenet', include_top=False)
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
base_model.trainable = False
# Freeze Inception's weights - we don't want to train these
for layer in base_model.layers:
    layer.trainable = False

# add a fully connected layer after Inception - we do want to train these
branch_2 = base_model.output
branch_2 = GlobalAveragePooling2D()(branch_2)
branch_2 = Dense(1024, activation='relu')(branch_2)

# merge the text input branch and the image input branch and add another fully connected layer
joint = concatenate([branch_2, branch_1])
joint = Dense(512, activation='relu')(joint)
joint = Dropout(0.5)(joint)
predictions = Dense(num_classes, activation='sigmoid')(joint)

full_model = Model(inputs=[base_model.input, text_inputs], outputs=[predictions])

full_model.compile(loss='categorical_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

print(full_model.summary())

# history = full_model.fit([X_train_image, X_train_text], y_train,
#                          epochs=epochs, batch_size=batch_size,
#                          verbose=1, validation_split=0.2, shuffle=True)

## checkpoints
checkpoints_path = r"D:/Projects/image_text_model/models"
checkpoints = ModelCheckpoint(checkpoints_path,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True,
                                verbose=1
                                )

history = full_model.fit_generator(
                        custom_img_txt_generator_class_train.image_text_generator(lines), 
                        epochs=epochs, 
                        steps_per_epoch=num_samples // batch_size,
                        validation_data=custom_img_txt_generator_class_test.image_text_generator(lines),
                        validation_steps=num_samples // batch_size,
                        verbose=1, 
                        shuffle=True,
                        use_multiprocessing=False,
                        callbacks=[checkpoints]
                        )

final_model_path = os.path.join(model_path, "model_final.hdf5")
full_model.save(final_model_path)