import numpy as np

a = np.loadtxt('C:\\test\\DCNN\\ranknetdata\\001.txt')
b = np.reshape(a,(500,1000))    
print b.shape
print b[0]