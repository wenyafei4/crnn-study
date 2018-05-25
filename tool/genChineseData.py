#coding:UTF-8
import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import re
#import Image
import numpy as np
import imghdr


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False		
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)
			
def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    print (len(imagePathList),len(labelList))
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    saveFile = open(outputPath,'w')
    cnt = 1
    for i in xrange(nSamples):   
       # imagePath = './recognition/'+''.join(imagePathList[i]).split()[0].replace('\n','').replace('\r\n','')
        imagePath = ''.join(imagePathList[i]).replace('\n','').replace('\r\n','')
        if len(labelList[i])==0:
            continue
        label = ''.join(labelList[i])
#	print imagePath
#	print label
	#exit(0)
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue	
		
        with open(imagePath, 'r') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        saveFile.write(imagePath + '###' + label + '\n')
        if cnt % 1000 == 0:
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
        print cnt
    nSamples = cnt-1
    print('Created dataset with %d samples' % nSamples)
	

if __name__ == '__main__':
    outputPath = "testComplex.txt"
    imgListFile = open("testComplex_imgList.txt_sort")
    labelListFile = open("testComplex_labelList.txt_sort")
    imagePathList = list(imgListFile)
    labelPathList = list(labelListFile)
    print len(labelPathList)
    
    labelList = []
    for labelfileName in labelPathList:
    #    print labelfileName
        labelFile = open(labelfileName.replace('\n','').replace('\r\n',''))
        label = list(labelFile)[0].replace('\n','').replace('\r\n','')
     #   print label
    #    targetString='布兰诺强哥'
    #    if ''.join(label)==targetString:
    #        print 'feiji'
    #        print labelFile
    #        break
        labelList.append(label)
    createDataset(outputPath, imagePathList, labelList)
    #pass
