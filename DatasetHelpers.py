import os




# Dictionary{classPath => list of files names}
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



# Dictionary{classPath => list of files names}
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