import os
import glob as gb
import numpy as np 
import cv2
import tensorflow as tf 
import decimal 
from keras.models import load_model
from keras import backend as K
from keras.models import Model

a = np.loadtxt('C:\\test\\DCNN\\ranknetdata\\testrugby.txt')
X_1 = np.reshape(a,(500,1000))    
b = np.loadtxt('C:\\test\\DCNN\\ranknetdata\\testrugby.txt')
X_2 = np.reshape(b,(500,1000))  
c=[]
d=[]
model = load_model('my_model.h5')
for index in range(500):
    test1=np.reshape(X_1[index],(1,1000)) 
    test2=np.reshape(X_2[index],(1,1000))  
    #print (model.predict([test1, test2], batch_size=1, verbose=0))
    # get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
    #                                   [model.layers[5].get_input_at(0)])
    # layer_output = get_3rd_layer_output([model.layers[0].input, 0])[0]
    layer_name = 'lambda_1'
    intermediate_layer_model = Model(input=model.input,
                                     output=model.get_layer(layer_name).get_input_at(0))
    intermediate_output = intermediate_layer_model.predict([test1, test2], batch_size=1, verbose=0)
    c.append(intermediate_output)
    d.append(1-intermediate_output) 
mat1 = np.array(c)
X_1 = np.reshape(mat1,(1,-1))   
np.savetxt("valresultnon.txt", X_1);  
mat2 = np.array(d)
X_2 = np.reshape(mat2,(1,-1)) 
np.savetxt("valresulthighlight.txt", X_2);  