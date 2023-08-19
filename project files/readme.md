from tensorflow.keras.preprocessing.image import ImageDataGenerator

#setting parameter for image data augmentation to the training data. 

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range = 0.1,
                                    zoom_range=0.1,
                                    horizontal_flip = True)


#image data augmentation to the testing data.

test_datagen = ImageDataGenerator(rescale=1./255)

train_folder="Desktop/Garbageclassification"
test_folder="Desktop/pattu/test"

# Generate the training and testing datasets using the flow_from_directory method
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(128,128),
    batch_size=64,
    class_mode='categorical'
)


test_generator = train_datagen.flow_from_directory(
    test_folder,
    target_size=(128,128),
    batch_size=64,
    class_mode='categorical'
)


#to define linear initializations import Sequential

from tensorflow.keras.models import Sequential

#To add Layers import Dense

from tensorflow.keras.layers import Dense

# to create a convolution kernel import Convolution2D 

from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import Convolution2D 

#Adding Max pooling Layer 

from tensorflow.keras.layers import MaxPooling2D  

#Adding Flatten Layer

from tensorflow.keras.layers import Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create a Sequential model
model = Sequential()

# Add the first Convolutional layer with 32 filters and a (3, 3) kernel size
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))

# Add a MaxPooling layer with a pool size of (2, 2)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output before feeding it to the Dense layers
model.add(Flatten())

# Add the first Dense layer with 150 units and ReLU activation
model.add(Dense(150, activation='relu'))

# Add the second Dense layer with 68 units and ReLU activation
model.add(Dense(68, activation='relu'))

# Add the final Dense layer with 6 units (6 classes for classification) and softmax activation
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

res = model.fit(
                          train_generator,
                          steps_per_epoch=2527//64, 
                          validation_steps=782//64,
                          epochs=30,
                          validation_data=test_generator)

model.save('Garbage1.h5')

import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('Garbage.h5')


from tensorflow.keras.preprocessing import image

img = image.load_img(r"glass9.jpg", target_size=(128,128))

x=image.img_to_array(img) #converting in to array format

x=np.expand_dims (x, axis=0) #changing its dimensions as per our requirement #img_data=preprocess_input(x)


#img_data.shape

a=np.argmax(model.predict(x), axis=1)


index = ['0','1','2','3','4','5']
result = str(index[a[0]])
result

index1 = ['CARDBOARD','GLASS','METAL','PAPER','PLASTIC','TRASH']
result1 = str(index1[a[0]])
print(result1)
img

