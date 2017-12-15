from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import warnings
import matplotlib.cbook
import pickle

from GMMs import *
from MFCC import *
from evaluation import *

"""
    *************************************************************************
    Main entry class of the project
    *************************************************************************
"""

# testing ot training, for serialization of GMM models
trainingMode = False #Flase means no training
minNumOfGaussians = 1
maxNumOfGaussians = 10

if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


    gmms = {}
    if trainingMode:
        count = 0
        classes = getTrainingDataMFCCs(folerPath='.\\training_data')
        for classLabel, classData in classes.items():
            gmms[classLabel] = calcaulteGMMForEachClass(np.array(classData),minNumOfGaussians, maxNumOfGaussians, count )
            count += 1
        pickle.dump(gmms, open('gmmModels/classesGMMs', 'wb'))
    else :
        gmms = pickle.load(open('gmmModels/classesGMMs', 'rb'))


    filesNames, testingDataMfcc = getTestingDataMFCCs(folerPath='.\\testing_data')

    print('Applying testing data ...')
    print('Result of each file ...')
    print()

    y_true = []
    y_pred = []
    filesNamesCounter = 0
    for classLbl, filesMfcc in testingDataMfcc.items():

        for fileMfcc in filesMfcc:
            predictedClassesProbability = []
            predictedClass = 0
            print("Testing file: " + filesNames[filesNamesCounter])
            filesNamesCounter += 1
            for label, gmm in gmms.items():
                gaussiansPrdections = gmm.score_samples(fileMfcc)
                finalScore = np.mean(gaussiansPrdections)
                predictedClassesProbability.append(finalScore)
                print(" probability for class '" + label.split("\\")[-1] + "' is: " + str(finalScore))

            predictedClass = predictedClassesProbability.index(max(predictedClassesProbability))
            print("predicted class is ' " + str(predictedClass) + "'" + " with probability: " + str(
                predictedClassesProbability[predictedClass]))
            currentTrueClass = classLbl.split("\\")[-1]
            print("    True class " + currentTrueClass + ", predicted class " + str(predictedClass))
            print("---------------------------------------------------------------")
            y_true.append(int(currentTrueClass))
            y_pred.append(predictedClass)

    print()
    print('EVALUATION OF SYSTEM')
    print("     Confusion Matrix:")
    confMatrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    numOfClasses = confMatrix.shape[0]
    print_confusion_matrix(confMatrix, [str(x) for x in range(numOfClasses)])
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print('     Precision is : ', precision)
    print('     Recall is : ', recall)
    print('     F-score is : ', fscore)



# https://gist.github.com/zachguo/10296432
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
