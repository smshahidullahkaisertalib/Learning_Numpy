import numpy as np 

arr = np.array([1,2,3, "Talib"], ndmin=10)
print(arr)

arr2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2[0,0])
print(arr2[0,2])
print(arr2[1,0])
print(arr2[1,1])
print(arr2[2,0])
print(arr2[2,2])