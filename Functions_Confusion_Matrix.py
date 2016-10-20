# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:30:16 2016

@author: wafaa
"""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, nbr_c, cmap, title='Confusion matrix'):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(nbr_c)
    plt.xticks(tick_marks, tick_marks[:]+1)
    plt.yticks(tick_marks, tick_marks[:]+1)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def estimate_confusion_matrix(True_y,Predict_y):
    # Matrice de confusion normalisée   
    CM = confusion_matrix(True_y,Predict_y)
    CM_N = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]*100     
    # La diagonale de la matrice de confusion
    CM_Classes = []    
    CM_Classes.append(np.array([CM_N[i, i] for i in range(CM_N.shape[0])]))
    return CM_N, CM_Classes  