# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:20:05 2020

@author: madhu
"""

from __future__ import print_function
import keras
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

modelsave_name = 'quarantine_face_mask.h5'

# restore the trained model (you can skip training the model by using just this)
new_model = keras.models.load_model(modelsave_name)

test = r'C:\Users\madhu\AppData\Roaming\SPB_Data\Face_Mask\test'

images = []

for file in list(os.listdir(test)):
    images.append(cv2.imread(test+'/'+file))
    


image = np.array(images)
testo = image[0]
print(testo)
m = new_model.predict(testo.reshape(-1,160,160,3)) == new_model.predict(testo.reshape(-1,160,160,3)).max()
plt.imshow(testo[:,:,::-1])

print(m[0])
print(np.array(['With Mask','Without Mask'])[m[0]])

print(1)