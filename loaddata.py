import numpy as np
import glob 
from PIL import Image
from keras.utils import np_utils

def loaddata(datadir,phase,sizeh, sizew,c):
    if phase == "train":
        traindir = "datasets/" + datadir + "/train/*"
        valdir   = "datasets/" + datadir + "/val/*"
        trainlists = glob.glob(traindir)
        vallists   = glob.glob(valdir)
        Xtrain = []
        Ytrain = []
        Xval   = []
        Yval   = []
        
        for i, trainlist in enumerate(trainlists):
            picturelists = glob.glob(trainlist + "/*")
            for picturelist in picturelists:
                img = Image.open(picturelist)
                data = img.resize((224, 224))
                print(data.size)
                print(picturelist)
                data = np.asarray(data)
                
                data = np.reshape(data, (sizeh,sizew,c))
                Xtrain.append(data)
                Ytrain.append(i)
        Xtrain = np.array(Xtrain)
        Ytrain = np.array(Ytrain)
        Xtrain = Xtrain.astype("float32")
        Xtrain = Xtrain / 255.0
        Ytrain = np_utils.to_categorical(Ytrain, len(trainlists))
        
        for i, vallist in enumerate(vallists):
            picturelists = glob.glob(vallist + "/*")
            for picturelist in picturelists:
                img = Image.open(picturelist)
                data = np.asarray(img)
                data = np.reshape(data,(sizeh,sizew,c))
                Xval.append(data)
                Yval.append(i)
        Xval = np.array(Xval)
        Yval = np.array(Yval)
        Xval = Xval.astype("float32")
        Xval = Xval / 255.0
        Yval = np_utils.to_categorical(Yval,len(vallists))
        return Xtrain, Ytrain, Xval, Yval 
    else:
        testdir = "datasets/" + datadir + "/test/*"
        testlists = glob.glob(testdir)
        Xtest = []
        Ytest = []
        for i, testlist in enumerate(testlists):
            picturelists = glob.glob(testlist + "/*")
            for picturelist in picturelists:
                img = Image.open(picturelist)
                data = np.asarray(img)
                data = np.reshape(data, (sizeh,sizew,c))
                Xtest.append(data)
                Ytest.append(i)
        Xtest = np.array(Xtest)
        Ytest = np.array(Ytest)
        Xtest = Xtest.astype("float32")
        Xtest = Xtest / 255.0
        Ytest = np_utils.to_categorical(Ytest, len(testlists))
        return Xtest, Ytest
