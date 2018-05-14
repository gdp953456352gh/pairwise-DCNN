import numpy

s=[]
array1=numpy.array([1,2,3,4,5,6])
array2=numpy.array([1,2,3,4,5,6])
array3=numpy.array([1,2,3,4,5,6])
s.append(array1)
s.append(array2)
s.append(array3)
array=numpy.array(s)
array4=numpy.reshape(array, len(s)*6)  
print type(array4)
print array4