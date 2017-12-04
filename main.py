from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from sklearn.mixture import GaussianMixture


def createListWithSpecificNumber(i,maxLen):
    a = []
    a = a + [i] * (maxLen - len(a))
    return a

def getMfccs(fileName):
    (samplingRate, signalSamples) = wav.read(fileName) #reading the file
    basicMfccfeatures = mfcc(signalSamples, samplingRate) # this will return (#frames, 13) features which are the 12 basic MFCCs + energy
    deltaMfccFeatures = delta(basicMfccfeatures, 2) #this will return (#frames, 13) which are the deltas
    doubleDeltaMFCCFeatures = delta(deltaMfccFeatures, 2) #this will return (#frames, 13) which are the dobule deltas

    allMfccs = np.concatenate((basicMfccfeatures, deltaMfccFeatures, doubleDeltaMFCCFeatures), axis=1) #this will return (#frames, 39) features
    return allMfccs


#this function will return every class, and the the wav files related to it
# as a dictonary (key is the path of the class, the value is list of files names)
def getTrainingData(folerPath='.\\training_data'):
    rootDir = folerPath

    classes = {}
    for dirName, subdirList, fileList in os.walk(rootDir):
        classes[dirName]  =[]
        for fileName in fileList:
            classes[dirName].append(fileName)

    classes.pop(rootDir) # remove the first element (root path)
    return classes

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def getTrainingDataMFCCs(folerPath='.\\training_data'):
    classesWithMfcc = {}
    labels = []
    data = []
    traingDataDictonary = getTrainingData(folerPath)
    counter = 0
    for classpath, filesName in traingDataDictonary.items():
        classesWithMfcc[counter] = []
        for fileName in filesName:
            filePath = classpath + '\\' + fileName
            # print('reading file %s ...' % filePath)
            currentFileMfccs = getMfccs(filePath)
            # print(currentFileMfccs)
            labels.extend(createListWithSpecificNumber(counter,len(currentFileMfccs)))
            data.extend(currentFileMfccs)
            # print(labels)
           # print(currentFileMfccs[0, :])  # print first row of mfcc
            classesWithMfcc[counter].append(currentFileMfccs)
        counter+=1
    return labels,data

if __name__ == '__main__':

    y_train,X_train = getTrainingDataMFCCs(folerPath='.\\training_data')
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_train)
    y_test = np.array(y_train)

    # print(x)
    # # print(labels)
    # print(len(data))
    # f = open('sehweil.txt','w')
    # f.write(''.join(str(e)+" " for e in labels))
    n_classes = len(np.unique(y_train))
    print(n_classes)

# Try GMMs using different types of covariances.
    estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
                       covariance_type=cov_type, max_iter=20, random_state=0))
                      for cov_type in ['spherical', 'diag', 'tied', 'full'])

    n_estimators = len(estimators)

    plt.figure(figsize=(3 * n_estimators // 2, 6))
    plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                        left=.01, right=.99)

    for index, (name, estimator) in enumerate(estimators.items()):

        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                        for i in range(n_classes)])

        # print(X_train)
        # Train the other parameters using the EM algorithm.
        estimator.fit(X_train)

        colors = ['navy', 'turquoise', 'darkorange']

        h = plt.subplot(2, n_estimators // 2, index + 1)
        make_ellipses(estimator, h)

        for n, color in enumerate(colors):
            data = X_train[y_train == n]
            plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
                        label=['0' '1' '2'])
        # Plot the test data with crosses
        for n, color in enumerate(colors):
            data = X_test[y_test == n]
            plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

        y_train_pred = estimator.predict(X_train)
        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
                 transform=h.transAxes)

        y_test_pred = estimator.predict(X_test)
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
                 transform=h.transAxes)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)
    plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
    plt.show()
