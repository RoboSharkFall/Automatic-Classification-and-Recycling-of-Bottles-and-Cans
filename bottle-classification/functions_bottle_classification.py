# Adding modules
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn import svm
import joblib
from tqdm import tqdm
import random
import pickle

def creat_data_sets(img_array):
    # Image width
    img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    IMG_W = int(640/2)
    # Image height
    IMG_H = int(480/2)
    data_set = []
    """Save images to data_set list"""
    # Iterate through the categories list
        # A directory that contains images for a specific category 
        # Index of the current category in the categories list
        # Names of individual images
    try:
    # Image upload
    #  img_array = cv2.imread('test.jpg') 
    # Image resizing
        new_array = cv2.resize(img_array, (IMG_W, IMG_H)) 
    # Adding an image and category index to the data_set list
        data_set.append(new_array) 
        # Create an empty list for X and Y
        X = []

        # Iterate through the data_set list, and extract data from it
        for features in data_set:
            # Adding features to X list.
            X.append(features)
            # Ignoring mistakes
        return X 
    except Exception as e:
     pass

def data(X_path,Y_path):
  """Open .pickle files, and restore X and Y lists"""
  # Open the X.pickle file
  pickle_in = open(X_path, "rb")
  X = pickle.load(pickle_in)
  # Open the Y.pickle file
  pickle_in = open(Y_path, "rb")
  Y = pickle.load(pickle_in)
  return X, Y

def sift(img):
  """Create SIFT method to exclude features, and return kp and des"""
  # Creating SIFT method
  sift = cv2.xfeatures2d.SIFT_create()
  # Determining the number of features and features
  kp, des = sift.detectAndCompute(img,None)
  return kp, des

def orb(img):
  """Create ORB method to exclude features, and return kp and des"""
  # Create ORB method
  orb = cv2.ORB_create()
  # Determining the number of features and features
  kp, des = orb.detectAndCompute(img,None)
  return kp, des

def surf(img):
  """Create SURF method to exclude features, and return kp and des"""
    # Create SURF method
  surf = cv2.xfeatures2d.SURF_create()
  # Determining the number of features and features
  kp, des = surf.detectAndCompute(img,None)
  return kp, des

def feature_number(feature):
  """Creating a list with the features of individual images, and returning list_data and ind"""
  # Creating a blank list ind
  ind = []
  # Create a blank list_data list
  list_data = []
  t0 = time()
  # Iteration from 0 to the total number of data in X
  for i in range(len(X)):
    # Execution of SIFT, SURF and ORB functions
    kp, des = feature(X[i])
    # If the number of features in that image is less than 20, the image does not qualify
    if len(kp) < 20:
      # Adding to ind list
      ind.append(i)
      continue
    # Forming a feature of equal size (equal number of data)
    des = des[0:20,:]
    # Formation of the obtained feature data in the form 1, len (des) * len (des [1])
    vector_data = des.reshape(1,len(des)*len(des[1]))
    # Adding vector_data to the list_data list
    list_data.append(vector_data)
  # List of names of feature extraction methods
  features = ['sift', 'surf', 'orb']
  print("Algorithm time: %0.3fs" % (time() - t0))
  return list_data, ind
    
def svm_parameters(X_train, y_train):
  """Finding parameters for model training and returning clf.best_estimator_"""
  t0 = time()
  # Parameters
  param_grid = {'C': [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.001, 0.01, 0.1], 
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
  # Parameter search function
  clf = GridSearchCV(
    svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
  clf = clf.fit(X_train, y_train)
  print("Parameter finding time: %0.3fs" % (time() - t0))
  return clf.best_estimator_

def svm_train(X_train, y_train):
  """Model training and returning clf"""
  t0 = time()
  # Creating an SVM classifier
  clf = svm.SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1e-8, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
  # Training of SVM classification model
  clf.fit(X_train, y_train)
  print("Model training time: %0.3fs" % (time() - t0))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
  return clf

def svm_test(clf, X_test, y_test):
  """Testing the model and returning y_pred"""
  t0 = time()
  # Testing of SVM classification model
  y_pred = clf.predict(X_test)
  # Model accuracy: what percentage is accurately classified data? (TP + TN) / (TP + TN + FP + FN)
  print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
  # Model precision: what is the percentage of positive identifications in a set of positively classified data? TP / (TP + FP)
  print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))
  # Model recall: what is the percentage of positive identifications in the set of all positive data? TP / (TP + FN)
  print("Recall:",metrics.recall_score(y_test, y_pred, average='micro'))
  # Table of results obtained
  print(classification_report(y_test, y_pred, target_names=categories))
  print("Model testing time: %0.3fs" % (time() - t0))
  return y_pred

def svm_save(clf, path):
  """Saving SVM model"""
  joblib.dump(clf, path)

def plot_gallery(images, titles, h, w, n_row=1, n_col=2):
  """Displays individual images, image categories, and default categories"""
  # Image window size
  plt.figure(figsize=(4 * n_col, 2 * n_row))
  # Image parameters
  plt.subplots_adjust(bottom=0, left=0.1, right=0.9, top=.95, hspace=.35)
  # Display a certain number of images
  for i in range(n_row * n_col):
    plt.subplot(n_row, n_col, i + 1)
    plt.imshow(images[i].reshape((w,h)))
    plt.title(titles[i], size=10)
    plt.xticks(())
    plt.yticks(())

def title(y_pred, y_test, target_names, i):
  """Extract the actual and default image categories, and return pred_name and true_name"""
  pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
  true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
  return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def classify_bottles(image):
  X = creat_data_sets(image)
  feature = orb
  """Creating a list with the features of individual images, and returning list_data and ind"""
  # Create a blank list_data list
  list_data = []
  t0 = time()
  # Iteration from 0 to the total number of data in X
    # Execution of SIFT, SURF and ORB functions
  kp, des = feature(X[0])
    # If the number of features in that image is less than 20, the image does not qualify
  print(len(kp))
  if len(kp) < 20:
    # Adding to ind list
     print("number of features is too low.")
     return
    # Forming a feature of equal size (equal number of data)
  des = des[0:20,:]
  # Formation of the obtained feature data in the form 1, len (des) * len (des [1])
  vector_data = des.reshape(1,len(des)*len(des[1]))
  print('len(des)',len(des),len(des[0]))
  # Adding vector_data to the list_data list
  list_data.append(vector_data)
  # Iterate through the list to delete data that didn't meet a sufficient number of features.
  # for i in sorted(ind, reverse=True):
  #   del labels[i]
  # Creating a vector in the form of len (labels), len (list_data [0] [0])
  data = np.array(list_data).reshape(1,len(list_data[0][0]))
  print('len(list_data[0][0])',len(list_data[0][0]))
  # Executing the svm_train() function
  clf = joblib.load("/home/sharkfall/projects/g_picking/Data/orb_trained_model.npy")
  # Executing the svm_test() function
  y_pred = clf.predict(data)
  print('y_pred',y_pred)
  return y_pred
  
