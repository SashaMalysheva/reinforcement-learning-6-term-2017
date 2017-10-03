import numpy as np
def hash(x, y):
    len = 4
    return x + y*len


def unhash(a):
    return (a  + 1 - (a %2)* 2) + (a//4)


print(unhash(0))
print(unhash(1))
print(unhash(2))
print(unhash(3))
