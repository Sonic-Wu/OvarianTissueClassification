from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import tensorflow.keras
import numpy as np
import os
#import env
import envi
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataload import DataGeneratorCNN
from sklearn.metrics import accuracy_score
import sklearn
from sklearn.preprocessing import StandardScaler
import pickle
import imageio
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

def loaddataCNN(tiles, maskfiles, ftirfile, maskpath, ftirpath, samples=None):
    os.chdir(ftirpath)
    ftirenvifile = envi.envi(ftirfile.replace("#", tiles[0]))
    dataXloc = np.empty((0, 3),dtype=np.int)
    dataY = np.empty((0),dtype=np.int)
    for j in range(len(tiles)):
        os.chdir(ftirpath)
        ftirenvifile = envi.envi(ftirfile.replace("#", str(tiles[j])))
        maskarray = np.empty((0, ftirenvifile.header.lines, ftirenvifile.header.samples))
        for maskfile in maskfiles:
            os.chdir(maskpath)
            mask = imageio.imread(maskfile.replace("#", str(tiles[j])), as_gray=True)
            maskarray = np.append(maskarray, np.reshape(mask, (1, mask.shape[0], mask.shape[1])), axis=0)
        datalocations = np.transpose(np.nonzero(maskarray))
        dataXseg = np.empty((len(datalocations),3),dtype=np.int)
        dataYseg = np.empty((len(datalocations)),dtype=np.int)
        for i in range(int(len(datalocations))):
            location = datalocations[i]
            dataYseg[i] = location[0]
            dataXseg[i,:] = [j,location[1],location[2]]
            if i % 25 == 0:
                print("\r" + str(int(float(i) / len(datalocations) * 100)) + "% loaded in tile " + str(tiles[j]), end="")
        print("\n")
        dataXloc = np.append(dataXloc,dataXseg,axis=0)
        dataY = np.append(dataY,dataYseg,axis=0)
    if samples != None:
        newdataXloc = np.empty((samples*len(maskfiles),3),dtype=np.int)
        newdataY = np.empty((samples*len(maskfiles)),dtype=np.int)
        for k in range(len(maskfiles)):
            locs = np.argwhere(dataY == k)
            np.random.shuffle(locs)
            #np.random.seed(1023)
            newdataY[k*samples:(k+1)*samples] = np.full((samples),k)
            newdataXloc[k*samples:(k+1)*samples,:] = np.reshape(dataXloc[locs[0:samples],:],(samples,3))
        dataXloc = newdataXloc
        dataY = newdataY

    return dataXloc, dataY

def compileimages(tiles, ftirpath, ftirfile):
    os.chdir(ftirpath)
    ftirenvifile = envi.envi(ftirfile.replace("#", str(tiles[0])))
    imcoord = []
    imagesarr = np.empty((0,ftirenvifile.header.bands))
    for tile in tiles:
        ftirenvifile = envi.envi(ftirfile.replace("#", str(tile)))
        ftirdata = ftirenvifile.loadall()
        imcoord.append([ftirdata.shape[1],ftirdata.shape[2]])
        ftirdata = np.moveaxis(ftirdata,0,-1)
        imagesarr = np.append(imagesarr,ftirdata.reshape((ftirdata.shape[0]*ftirdata.shape[1],ftirdata.shape[2])),axis=0)
    #images = np.reshape(images,(len(tiles),ftirenvifile.header.bands,ftirenvifile.header.lines*ftirenvifile.header.samples))
    #images = np.moveaxis(images,1,-1)
    #images = np.reshape(images,(len(tiles)*ftirenvifile.header.lines*ftirenvifile.header.samples,ftirenvifile.header.bands)))


    images = []
    idx = 0
    for coordinates in imcoord:
        images.append(np.reshape(np.moveaxis(imagesarr[idx:idx+coordinates[0]*coordinates[1],:],-1,0),(imagesarr.shape[-1],coordinates[0],coordinates[1])))
        idx = idx + coordinates[0]*coordinates[1]
    return images

def buildmodel(bands, imsize, num_categories):
    ###########################################################################
    #nor = initializers.RandomNormal(stddev=0.02, seed=100)
    #nor = initializers.RandomNormal(stddev=.005, seed=42)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1 + 2 * imsize, 1 + 2 * imsize, bands) ) )
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ##########################################################################
    model.add(Conv2D(32, (3, 3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ###########################################################################
    model.add(Conv2D(64, (3, 3) ))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ###########################################################################
    #model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(64))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    ##########################################################################
    model.add(Dense(2))
    model.add(Activation('softmax'))

    ##########################################################################
    # initiate AdaDelta optimizer
    opt = tensorflow.keras.optimizers.Adam()

    ##########################################################################
    # Let's train the model using Adadelta
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

def train_predict(iter):
    scores = []
    for i in range(iter):

        #maskpath = "/project/reddy/Chalapathi/CNN/masks/"
        #ftirpath = "/project/reddy/Chalapathi/CNN/singletile/"
        #homepath = /home/cgajjela/CNN
        
        maskpath = "/home/projects/Projects/Cervix_data/cervix_masks"
        #ftirpath = "/home/projects/Projects/Cervix_data/envi_dc/"
        ftirpath = "/home/projects/Projects/Cervix_data/envi/"
        
        tiles = ["A1","A2","A3","A4","A5",
                  "B1","B2","B3","B4","B5",
                  "C1","C2","C3","C4","C5",
                  "D1","D2","D3","D4","D5",
                  "E1","E2","E3","E4","E5",
                  "F1","F2","F3","F4","F5"
                  ,"G1","G2","G3","G4","G5"]
        #tiles = ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5","C1","C2","C3","C4","C5","D1","D2","D3","E1","E2","E3","F1","F2","F3"]

        """tiles = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10",
                  "B1","B2","B3","B4","B5","B6","B7","B8","B9","B10",
                  "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
                  "D1","D2","D3","E1","E2","E3","F1","F2","F3"]"""
        #tiles = ["A1","A2","A3","A4","B1","B2","B3","B4"] --75.8
        #tiles = ["A1","A2","A3","A4","B1","B2","B3","B4","C1","C3","D1","D2"]
        #tiles = ["H1","H2","H3","H4","H5"]
        #tiles = ["H2","H3","H4"]
        #tiles = ["C1","C2","C3","C4","F1","F2","F3","F4","F5","J2","J3","J4","J5","I1","I2","I3","I4","I5","E1","E2","E3","E4","E5"]
        #tiles = ["rows-c-j-bkg-norm"]
        #maskfiles = ["epith-train-rows-c-j.png", "stroma-train-rows-c-j.png", "necrosis-train-rows-c-j.png"]
        #maskfiles = ["epith-train-rows-c-j.png", "stroma-train-rows-c-j.png"]
        maskfiles = ["epithelium-#-cervix.png", "stroma-#-cervix.png"]
        #ftirfile =  "#_dcN"
        ftirfile =  "#_1660norm"
        

        #savedirectory = "/project/reddy/Chalapathi/CNN/"
        savedirectory = "/home/projects/Projects/Cervix_data/results/train-test-3iter-400k-epoch4/"
        
        
        cores= "C-J-32"
        #savefile = "CNNmiragefinal-fullcores_C-J_32-epoch8_400kbin" +cores+ str(i) + ".h5"
        savefile = "CNNmiragefinal-fullcores_C-J_32-epoch8_400kbin_1664-5band" +cores+ str(i) + ".h5"

        imsize = 16
        num_categories = len(maskfiles)

        X_train_loc, y_train = loaddataCNN(tiles, maskfiles, ftirfile, maskpath, ftirpath,400000)
        locs = np.argwhere(y_train == 0)
        epi = len(locs)
        print("epi_train: ", epi)
        locs = np.argwhere(y_train == 1)
        epi = len(locs)
        print("stro_train: ", epi)
        
        #X_train_loc, X_val_loc, y_train, y_val = train_test_split(dataXloc, dataY, test_size = 0.0, random_state = 42)


        #os.chdir(ftirpath)
        #ftirenvifile = envi.envi(ftirfile.replace("#", str(tiles[0])))
        #imcoord = []
        #imagesarr = np.empty((0,ftirenvifile.header.bands))
        #for tile in tiles:
        #    ftirenvifile = envi.envi(ftirfile.replace("#", str(tile)))
        #    ftirdata = ftirenvifile.loadall()
        #    imcoord.append([ftirdata.shape[1],ftirdata.shape[2]])
        #    ftirdata = np.moveaxis(ftirdata,0,-1)
        #    imagesarr = np.append(imagesarr,ftirdata.reshape((ftirdata.shape[0]*ftirdata.shape[1],ftirdata.shape[2])),axis=0)
        ##images = np.reshape(images,(len(tiles),ftirenvifile.header.bands,ftirenvifile.header.lines*ftirenvifile.header.samples))
        ##images = np.moveaxis(images,1,-1)
        ##images = np.reshape(images,(len(tiles)*ftirenvifile.header.lines*ftirenvifile.header.samples,ftirenvifile.header.bands)))


        #images = []
        #idx = 0
        #for coordinates in imcoord:
        #    images.append(np.reshape(np.moveaxis(imagesarr[idx:idx+coordinates[0]*coordinates[1],:],-1,0),(imagesarr.shape[-1],coordinates[0],coordinates[1])))
        #    idx = idx + coordinates[0]*coordinates[1]

        images = compileimages(tiles, ftirpath, ftirfile)
        dg = DataGeneratorCNN(images, X_train_loc, y_train, imsize, num_categories, batch_size=128)

        model = buildmodel(images[0].shape[0], imsize, num_categories)
        print("Iteration "+ str(i))
        model.fit_generator(dg,epochs=4,verbose=1)
        
        os.chdir(savedirectory)
        model.save(savefile)

        os.chdir(ftirpath)
        #tiles = ["A5","A6","A7","A8","A9","A10"]
        #tiles = ["A5","A6","A7","A8","A9","A10","B4","B5","B6","B7","B8","B9","B10","C1","C2","C3","C4","C6","C7","C8","C9","C10"] --75.8
        #tiles = ["A5","A6","A7","A8","A9","A10","B4","B5","B6","B7","B8","B9","B10","C2","C6","C7","C8","C9","C10","D3","D4","D5","D8","D9","D10","E1","E2","E3","E4","E5"]
        #tiles = ["H6","H7","H8","H9","H10"]
        #tiles = ["C6","C7","C8","C9","C10","F6","F7","F8","F9","F10","J6","J7","J8","J9","J10","I6","I7","I8","I9","I10","E6","E7","E8","E9","E10"]
        #tiles = ["rows-c-j-bkg-norm"]
        #tiles = ["A6","A7","A8","A9","A10","B6","B7","B8","B9","B10","C6","C7","C8","C9","C10"]
        #tiles = ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5","C1","C2","C3","C4","C5","D1","D2","D3","E1","E2","E3","F1","F2","F3"]
        tiles = ["A6","A7","A8","A9","A10","B6","B7","B8","B9","B10","C6","C7","C8","C9","C10","D6","D7","D8","D9","D10","E6","E7","E8","E9","E10","G6","G7","G8","G9","G10"]
        maskfiles = ["epithelium-#-cervix.png", "stroma-#-cervix.png"]
        #maskfiles = ["epith-test-rows-c-j.png", "stroma-test-rows-c-j.png", "necrosis-test-rows-c-j.png"]
        #maskfiles = ["epithelium-mask-half-tile-#-test.png", "stroma-mask-half-tile-#-test.png"]
        X_val_loc, y_val = loaddataCNN(tiles, maskfiles, ftirfile, maskpath, ftirpath)
        images = compileimages(tiles, ftirpath, ftirfile)
        dgpredict = DataGeneratorCNN(images, X_val_loc, y_val, imsize, num_categories, batch_size=128, random=False)

        predprob = model.predict_generator(dgpredict, verbose=1)
        pred = predprob.argmax(axis=-1)
        locs = np.argwhere(y_val == 0)
        epi = len(locs)
        print("epi: ", epi)
        a1 = np.zeros(epi)
        b1 = np.zeros(epi)
        c1 = np.zeros(epi)
        for j in range(epi):
            a1[j] = y_val[locs[j,0]]
            #b1[j] = rfc_predict[locs[j,0]]
            c1[j] = pred[locs[j,0]] 
        print(' CNN epi accuracy: ', accuracy_score(a1, c1))
        
        locs = np.argwhere(y_val == 1)
        epi = len(locs)
        print("stro: ", epi)
        a1 = np.zeros(epi)
        b1 = np.zeros(epi)
        c1 = np.zeros(epi)
        for j in range(epi):
            a1[j] = y_val[locs[j,0]]
            #b1[j] = rfc_predict[locs[j,0]]
            c1[j] = pred[locs[j,0]] 
        print(' CNN stro accuracy: ', accuracy_score(a1, c1))
        
        os.chdir(savedirectory)        
        np.savetxt("ftiroldmodel-65cores-test-epoch8_400kbin_1660norm"+str(i)+"prob.csv",predprob,delimiter=',')
        np.savetxt("classesoldmodel-65cores-test-epoch8_400kbin_1660norm.csv",y_val,delimiter=',')
        score = accuracy_score(y_val,pred)
        scores.append(score)
        print("Score: " + str(score))
        print("Scores: " + str(scores))
    return scores

scores = train_predict(3)
print(scores)

savedirectory = "/home/projects/Projects/Cervix_data/results/train-test-3iter-400k-epoch4/"
os.chdir(savedirectory)
save = open("mirageresults-cerv-65cores-test-epoch8_400kbin_1660norm.txt",'w')
save.write(str(scores))


