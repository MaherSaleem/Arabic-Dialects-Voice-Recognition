import os

"""
    ******************************************************************************
    This class contains methods to handle loading of training and testing data-sets
    ******************************************************************************
"""


"""
    This function will return every class, and the the wav files related to it
    as a dictionary (key is the path of the class, the value is list of files names)
"""
def getTrainingData(folerPath='.\\training_data'):
    rootDir = folerPath

    classes = {}
    for dirName, subdirList, fileList in os.walk(rootDir):
        classes[dirName] = []
        for fileName in fileList:
            classes[dirName].append(fileName)

    # remove the first element (root path)
    classes.pop(rootDir)
    return classes



"""
    This function will return every class, and the the wav files related to it
    as a dictionary (key is the path of the class, the value is list of files names)
"""
def getTestingData(folerPath='.\\testing_data'):
    rootDir = folerPath

    classes = {}
    for dirName, subdirList, fileList in os.walk(rootDir):
        classes[dirName] = []
        for fileName in fileList:
            classes[dirName].append(fileName)

    # remove the first element (root path)
    classes.pop(rootDir)
    return classes