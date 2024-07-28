import os
import shutil
 
 
def splitImg2Category(dataDir, resDir, dataList):
    '''
    归类图像到不同目录中
    '''
    infos = [x.strip().split(' ')[0] for x in open(dataList).readlines()]
    for one_pic in os.listdir(dataDir):
        if one_pic.split('.jpg')[0] not in infos:
            continue
        one_path=dataDir+one_pic
        oneDir=resDir+'_'.join(one_pic.split('_')[:-1])
        if not os.path.exists(oneDir):
            os.makedirs(oneDir)
        shutil.copy(one_path,os.path.join(oneDir,one_pic))

dataDir = "cat_dog_classify/images/"
resDir = "cat_dog_classify/data/test/"
dataList ='cat_dog_classify/data/test.txt'
splitImg2Category(dataDir,resDir,dataList)