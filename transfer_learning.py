

# The following code is a CNN that will train on the MNIST dataset  
# Then will use transfer learning to train a DNN to classifiy
# a handwritten data set of 5 characters (ie A, B, C, D, E) 


# Seed code From https://keras.io/examples/mnist_transfer_cnn/


#%%
from __future__ import print_function
import datetime
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,  Conv2D, MaxPooling2D
from keras import backend as K

#%%
now = datetime.datetime.now
#batch_size = 128
num_classes = 5
#epochs = 5
img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)



#%%
def train_model(model, train, test, num_classes, epochs, batch_size, patience):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    t = now()
    es = keras.callbacks.EarlyStopping(min_delta=0.001, patience=patience)
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[es],
              verbose=1,
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return history



#%%
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]
x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5



#%%
feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('swish'),
    Conv2D(filters, kernel_size),
    Activation('swish'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]


#%%
classification_layers = [
    Dense(128),
    Activation('swish'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

# create complete model
model = Sequential(feature_layers + classification_layers)

#%%
from matplotlib import pyplot as plt
import numpy as np
def plot_training_curves(history, title = None):
  '''Plot the training curves for loss and accuracy given a model history
  '''
  # find the minimum loss epoch
  minimum = np.min(history.history['val_loss'])
  min_loc = np.where(minimum == history.history['val_loss'])[0]
  # get the vline y-min and y-max
  loss_min, loss_max = (min(history.history['val_loss'] + history.history['loss']),
                        max(history.history['val_loss'] + history.history['loss']))
  acc_min, acc_max = (min(history.history['val_accuracy'] + history.history['accuracy']),
                      max(history.history['val_accuracy'] + history.history['accuracy']))
  # create figure
  fig, ax = plt.subplots(ncols = 2, figsize = (15, 7))
  fig.suptitle(title)
  index = np.arange(1, len(history.history['accuracy']) + 1)
  # plot the loss and validation loss
  ax[0].plot(index, history.history['loss'], label = 'loss')
  ax[0].plot(index, history.history['val_loss'], label = 'val_loss')
  ax[0].vlines(min_loc + 1, loss_min, loss_max, label = 'min_loss_location')
  ax[0].set_title('Loss')
  ax[0].set_ylabel('Loss')
  ax[0].set_xlabel('Epochs')
  ax[0].legend()
  # plot the accuracy and validation accuracy
  ax[1].plot(index, history.history['accuracy'], label = 'accuracy')
  ax[1].plot(index, history.history['val_accuracy'], label = 'val_accuracy')
  ax[1].vlines(min_loc + 1, acc_min, acc_max, label = 'min_loss_location')
  ax[1].set_title('Accuracy')
  ax[1].set_ylabel('Accuracy')
  ax[1].set_xlabel('Epochs')
  ax[1].legend()
  plt.savefig(title+'_plots.png')
  plt.show()

#%%
# train model for 5-digit classification [0..4]
hist_lt5=train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), 5, 50, 128, 5)

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False

#%%
plot_training_curves(hist_lt5, "MNIST_Training_0-4-swish")


print('''
I printed 13x17 grid paper, filled them in with letters A-E then cropped the white space surrounding the grid in a Photo editor. 
This yielded 221 examples for each letter. The following scripts will load the images with each letter grid, split into columns, then break the columns
into individual images and save them to a dir with the letter as the first character of the file name, then row, and index.
''')

#%%
from PIL import Image

def load_image(file):
    return Image.open("./letter_photos/" + str(file) + "_raw.jpeg")

row_splits = 17
col_splits = 13

def find_split_size(size):
    col_spl = (size[0]//col_splits) 
    row_spl= (size[1]//row_splits) + 0.5
    return [row_spl, col_spl]

def row_split_coords(img):
    (width, height) = img.size
    (row_splt, col_splt) = find_split_size((width, height))
    y_split_coords=[]
    for i in range(0,17):
        x = 0
        y = row_splt*i 
        w = width
        h = row_splt+y
        y_split_coords.append([x, y, w, h])
    return y_split_coords

def col_split_coords(img):
    (width, height) = img.size
    (row_splt, col_splt) = find_split_size((width, height))
    x_split_coords=[]
    for i in range(0,13):
        x = col_splt * i
        y = 0
        w = col_splt + x
        h = height+y
        x_split_coords.append([x, y, w, h])
    return x_split_coords

def split_rows(letter, img):
    row_coords=row_split_coords(img)
    rows = [img.crop(coord) for coord in row_coords]
    for i,row in enumerate(rows):
        #row_img=img.crop(coord)
        split_cols_save(letter, i, row)

def split_cols_save(letter, row_num, img):
    col_coords=col_split_coords(img)
    fin_img = [img.crop(coord) for coord in col_coords]
    for i,img in enumerate(fin_img):
        #img.show()
        resized_img = img.resize((28,28))
        resized_img.save("./processed_letters/" + str(letter) + "_" + str(row_num) + str(i) + ".png")
    #return fin_img

def process_img(letter):
    raw_img = load_image(letter)
    split_rows(letter,raw_img)


letters = ["A","B","C","D","E"]

for letter in letters:
    process_img(letter)


print(
'''
This will load all of the photos in the dir with the data, convert to grayscale then numpy array. 
The labels are grabbed from the first character of the file name and saved to a list. The labels are
OHE prior to training.
''')

# %%
from PIL import Image
import os, os.path

imgs = []
labels=[]
path = "/Users/jason/Documents/research/transfer_learning/processed_letters/"
valid_images = [".png"]
for f in os.listdir(path):
    files=[]
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    labels.append(f[:1])
    imgs.append(Image.open(os.path.join(path,f)).convert('LA'))

#%%
from keras.utils import img_to_array

np_imgs = [img_to_array(img)[:,:,-2] for img in imgs]

def ohe_letters(lab):
    rv = 0
    if lab == 'B':
        rv += 1
    elif lab == 'C':
        rv += 2
    elif lab == 'D':
        rv += 3
    elif lab == 'E':
        rv += 4
    return rv


ohe_label = [ohe_letters(x) for x in labels]

from sklearn.model_selection import train_test_split
x_trn, x_tst, y_trn, y_tst = train_test_split(np_imgs, ohe_label, test_size=0.2, random_state=42)


#%%

# transfer: train dense layers for new classification task [5..9]
letter_training=train_model(model,
            (np.asarray(x_trn), y_trn),
            (np.asarray(x_tst), y_tst), num_classes, 1000, 128, 100)


#%%

plot_training_curves(letter_training, "Letter Training - 128 batch - 0-4 MNIST - swish")



print("""
I tried a number of combinations to improve the accuracy and loss metrics on the self-hand written dataset. 
The largest increase in performance was had by changing the optimizer from adadelta to adam. This drammatically increased performance.
The adadelta optimizer was yielding around 75% accuracy for the self-hand written dataset, and when changed the accuracy  was ~98-99%.
I also changed the activation functions from relu to swish, which was similar in performance.

It did so well on the hand written dataset, it seemed like I was doing something wrong. The images didn't have a great deal done to them in
terms of preprocessing. The images had different lighting, they weren't centered, and some of them had some lines around the edges from the
grid in the graph paper that I had used. Even still, it seemed like the model performed well.

""")
# %%
