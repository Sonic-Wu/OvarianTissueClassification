import numpy as np
import os
import envi
import matplotlib.pyplot as plt
#import keras
#from dataload import DataGeneratorCNN
from sklearn.metrics import accuracy_score
import imageio
import pandas
import pickle
from sklearn.preprocessing import StandardScaler
import imageio
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def loaddataCNN(tiles, maskfiles, ftirfile, maskpath, ftirpath):
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
    return dataXloc, dataY

#maskpath = "/project/reddy/Chalapathi/CNN/masks/randomforest_mask/"
#maskpath = "/project/reddy/Chalapathi/CNN/masks/"
#ftirpath = "/project/reddy/Chalapathi/CNN/singletile/"
#tiles = ["rows-c-j-bkg-norm"]
#tiles = ["A1","A2","A3","A4","A5","A6","A7","A8","A9",
#                  "B1","B2","B3","B4","B5","B6","B7","B8","B9",
#                  "C1","C2","C3","C5","C6","C7","C8","C9","C10",
#                  "D1","D2","D3","E1","E2","E3","F1","F2","F3"]
#tiles = ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5","C1","C2","C3","C4","C5","D1","D2","D3","E1","E2","E3","F1","F2","F3"]
"""tiles = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10",
                  "B1","B2","B3","B4","B5","B6","B7","B8","B9","B10",
                  "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
                  "D1","D2","D3","D4","D5","D6","D7","D8","D9","D10",
                  "E1","E2","E3","E4","E5","E6","E7","E8","E9","E10",
                  "F1","F2","F3","F4","F5",
                  "G1","G2","G3","G4","G5","G6","G7","G8","G9","G10"]"""
tiles = ["A6","A7","A8","A9","A10","B6","B7","B8","B9","B10","C6","C7","C8","C9","C10","D6","D7","D8","D9","D10","E6","E7","E8","E9","E10","G6","G7","G8","G9","G10"]
#maskpath = "/home/projects/Projects/CNN/masks/singletile_oldmask/"
#ftirpath = "/home/projects/Projects/CNN/singletile/"
#maskpath = "/home/projects/Projects/Cervix_data/tissue_mask/"
maskpath = "/home/projects/Projects/Cervix_data/cervix_masks"
ftirpath = "/home/projects/Projects/Cervix_data/envi/"
#ftirpath = "/home/projects/Projects/Cervix_data/envi_dc/"
#tiles = ["A1","A2","A3","A4","A5","A5","A6","A7","A8","A9","A10","B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","C2","C5","G1","I10"]
#maskfiles = ["epith-test-rows-c-j.png", "stroma-test-rows-c-j.png"]
#maskfiles = ["epith-train-rows-c-j.png", "stroma-train-rows-c-j.png", "lymph-train-rows-c-j.png"]
#maskfiles = ["epith-test-rows-c-j.png", "stroma-test-rows-c-j.png", "necrosis-test-rows-c-j.png"]
#maskfiles = ["rows-c-j-tissue-mask.png", "rows-c-j-tissue-mask.png"]
#maskfiles = ["#_tm.png", "#_tm.png"]
#maskfiles = ["rows-c-j-bkg-train-epith.png", "rows-c-j-bkg-train-stroma.png"]
#ftirfile =  "#"
maskfiles = ["epithelium-#-cervix.png", "stroma-#-cervix.png"]
#maskfiles = ["#_tm.png", "#_tm.png"]
ftirfile =  "#_1660norm"
#ftirfile =  "#_dcN"

#savedirectory = "/project/reddy/Chalapathi/CNN/singletile/4iter/"
#savedirectory = "/project/reddy/Chalapathi/CNN/4_epochs/final_masks/August 10/nodropout/"
#savedirectory = "/home/projects/Projects/CNN/singletile/"

#predpath = "/project/reddy/Chalapathi/CNN/singletile/4iter/"
predpath = "/home/projects/Projects/Cervix_data/results/train-test-3iter-400k-epoch4/"
#predpath = "/project/reddy/Chalapathi/CNN/singletile/"
#savedirectory = "/home/projects/Projects/Cervix_data/results_cores/39cores_1/dcN/"
#savedirectory = "/home/projects/Projects/Cervix_data/results_cores/39cores_1/1660norm/"
savedirectory = "/home/projects/Projects/Cervix_data/results_cores/65cores/"
#predfile = "ftir_oldmasks_25mil_C-J_32-epoch8_400kbin0prob.csv"
predfile = "ftiroldmodel-65cores-test-epoch8_400kbin_1660norm3prob.csv"
#ftiroldmodel-39cores-epoch8_train-epoch8_400kbin_1660norm0prob.csv
#predfile = "rf_fultissue.csv"

#labelcolors = {1:[1,0,0],2:[0,1,0]}
#labelcolors = {1:[1,0,0],2:[0,1,0],3:[0,0,1]}
os.chdir(predpath)
probs = pandas.read_csv(predfile, header=None).to_numpy()
#probs = probs * 255
pred = probs[:,0]
#pred1 = probs[:,1]
#pred2 = pred1
#pred = probs.argmax(axis=-1)
print(probs.shape)
print(pred.shape)
#print(np.amax(pred2))

for i in range(len(pred)):
    if(pred[i] > 0.8):
        pred[i] = 1
    else:
      if((pred[i] < 0.8) and (pred[i] > 0.2)):
        pred[i] = 3
      else:
        pred[i] = 2

"""for i in range(556212248):
    if(probs[i,1] <128):
        pred1[i] = 0
        
for i in range(556212248):
    if((pred1[i] == 255) or (pred[i] == 255)):
        pred2[i] = 0
    else:
        #c = pred1[i] - pred[i]
        #pred2[i] = np.abs(c)
        pred2[i] = 255 """
    

X_val_loc, y_val = loaddataCNN(tiles, maskfiles, ftirfile, maskpath, ftirpath)

print(X_val_loc.shape, y_val.shape)
"""locs = np.argwhere(y_val == 0)
epi = len(locs)
print(epi)
a1 = np.zeros(epi)
b1 = np.zeros(epi)
c1 = np.zeros(epi)
for j in range(epi):
    a1[j] = y_val[locs[j,0]]
    #b1[j] = rfc_predict[locs[j,0]]
    c1[j] = pred[locs[j,0]] 
print(' RF epi accuracy: ', accuracy_score(a1, c1))

locs = np.argwhere(y_val == 1)
epi = len(locs)
print(epi)
a1 = np.zeros(epi)
b1 = np.zeros(epi)
c1 = np.zeros(epi)
for j in range(epi):
    a1[j] = y_val[locs[j,0]]
    #b1[j] = rfc_predict[locs[j,0]]
    c1[j] = pred[locs[j,0]] 
print(' RF stro accuracy: ', accuracy_score(a1, c1))
locs = np.argwhere(y_val == 2)
epi = len(locs)
print("lymph: ", epi)
a1 = np.zeros(epi)
b1 = np.zeros(epi)
c1 = np.zeros(epi)
for j in range(epi):
    a1[j] = y_val[locs[j,0]]
    #b1[j] = rfc_predict[locs[j,0]]
    c1[j] = pred[locs[j,0]] 
print(' CNN lymph accuracy: ', accuracy_score(a1, c1))"""

#labelcolors = {0:[1,0,0],1:[0,1,0]}
labelcolors = {1:[1,0,0],2:[0,1,0],3:[0,0,1]}

for j in range(len(tiles)):
    os.chdir(ftirpath)
    envifile = envi.envi(ftirfile.replace("#", str(tiles[j])))
    #tileann = np.zeros((envifile.header.lines, envifile.header.samples, 3),'uint8')
    tileann = np.zeros((envifile.header.lines, envifile.header.samples, 3))
    for i in range(X_val_loc.shape[0]):
        if X_val_loc[i, 0] == j:
            #tileann[X_val_loc[i, 1], X_val_loc[i, 2], :] = labelcolors[y_val[i]]
            tileann[X_val_loc[i, 1], X_val_loc[i, 2], :] = labelcolors[pred[i]]
            #tileann[X_val_loc[i, 1], X_val_loc[i, 2], 0] = pred[i]
            #tileann[X_val_loc[i, 1], X_val_loc[i, 2], 1] = pred1[i]   
            #tileann[X_val_loc[i, 1], X_val_loc[i, 2], 2] = pred2[i]            

    os.chdir(savedirectory)
    img = Image.fromarray(tileann,"RGB")
    #img.save('rgb-result-fulltissue-rg-1.png')
    #img.save('fulltissue-rg-1.png')
    #imageio.imwrite("result-epoch8_100k-tile-test-" + str(tiles[j]) + ".png", tileann)
    #imageio.imwrite("rgb-result-fulltissue-" + str(tiles[j]) + ".png", tileann)
#    for i in range(X_val_loc.shape[0]):
#        if X_val_loc[i, 0] == j:
#            tileann[X_val_loc[i, 1], X_val_loc[i, 2], :] = labelcolors[y_val[i]]
    imageio.imwrite("cnn-80thres-test-65cores-1660norm-" + str(tiles[j]) + ".png", tileann)