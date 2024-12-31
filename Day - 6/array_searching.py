import numpy as np

arr = np.array([1,2,3,4,5,6])
print(np.where(arr%2 == 0))
print(np.where(arr%2 != 0))

arr2 = np.array([1,3,5,4,2,7,6,8])
print(f'3 is at index {np.where(arr2 == 3)}')
print(f"3 should be at index {np.searchsorted(arr2, 3)}")

print(f'6 is at index {np.where(arr2 == 6)}')
print(f"6 should be at index {np.searchsorted(arr2, 6)}")



arr3 = np.array([1,2,3,4,5,6,7,8])

print(f"6 should be at index {np.searchsorted(arr3,[2,5])}")
