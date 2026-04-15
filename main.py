import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import Adam
import glob
import argparse
from PIL import Image
import numpy as np 
from keras.utils import np_utils
from model.vgg16 import modelVGG16
from plot_history import plot_history_acc, plot_history_loss
import os
import json
from loaddata import loaddata
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', default="bike", help="./dataset/dirname/train/...")
    parser.add_argument('--phase', default="train", help="train?test?, def=train")
    parser.add_argument('--picsize', type=int, default=48, help="picturesize, def=48")
    parser.add_argument('--chanel', type=int, default=3, help="gray=1, RGB=3, def=1")
    parser.add_argument('--numcategory', type=int, default=3, help="category, def=7")
    parser.add_argument('--epoch', type=int, default=20, help="def=200")
    parser.add_argument('--batch', type=int, default=4, help="def=64")
    parser.add_argument('--opt', default="Adam")
    #parser.add_argument('--name')
    args = parser.parse_args()
    if args.phase == "train":
        #os.mkdir("./sample")
        Xtrain, Ytrain, Xval, Yval = loaddata(args.dirname, args.phase,
                                              224, 224, args.chanel)
        datagen = ImageDataGenerator(rotation_range=35,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     horizontal_flip=True)
        datagen.fit(Xtrain)
        print(len(Xtrain))
        model = modelVGG16(224, 
                           224,
                           args.chanel,  
                           args.numcategory)
        model.compile(loss="categorical_crossentropy",
                      optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      metrics=['accuracy'])
        """
        cp = ModelCheckpoint(filepath="./sample/weights{epoch:04d}.h5",
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode="min",
                             period=1)
        """
        modelhistory = model.fit_generator(datagen.flow(Xtrain, Ytrain, 
                                 batch_size=args.batch),
                                 steps_per_epoch = len(Xtrain)/args.batch,
                                 epochs=args.epoch,
                                 verbose=1,
                                 validation_data=(Xval, Yval),
                                 shuffle=True)#,
                                 #callbacks=[cp])
        plot_history_loss(modelhistory, "sample")
        plot_history_acc(model.history, "sample")
        with open("sample/history.json","w") as f:
            json.dump(modelhistory.history, f)
        model.save('test.h5')
    else:
        Xtest, Ytest = loaddata(args.dirname, args.phase)
        print(Xtest[0])
    
