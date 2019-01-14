import os
import pickle
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import cv2

def showImages(imgList):
    plt.figure(1)
    l = len(imgList)
    for i in range(l):
        plt.subplot(1,l,i+1)
        plt.imshow(imgList[i])
    plt.show()
    
def filelist(folder,nameFilter=None):
    files = list()
    if not nameFilter:
        files = natsorted([folder+os.sep+x for x in os.listdir(folder)])
    else:    
        for x in os.listdir(folder):
            if nameFilter in x:
                files.append(folder+os.sep+x)
        files = natsorted(files)

    return files

def path2class(path):
    name = os.path.basename(path).split('.')[0]
    cl = int(name.split('_')[1])
    return cl

def saveMatcher(outPath,matcher):
    with open(outPath,'wb') as f:
        pickle.dump(matcher,f)

def loadMatcher(inPath):
    with open(inPath,'rb') as f:
        M = pickle.load(f)
    return M

def topN(matches,n = 5):
    #### returns the top N predicted classes for the input matches
    #### topN[0] would be the most probable class

    vals,counts = np.unique(matches,return_counts=True)
    ordPos = np.flip(np.argsort(counts),axis = 0)
    vals = vals[ordPos]
    if n > len(vals):
        topN = vals
    else:
        topN = vals[0:n]

    return topN


