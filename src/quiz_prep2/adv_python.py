# Mutable vs Immutable Objects

# Immutable Objects

#Integers 
x = 10 
y = x 
x = 15
print(x) # 15
print(y) # 10   This is because a y is pointing to the same memory location as done previously by x , but x is now pointing to a different memory location 15 , Immutable actually means that we cannot change the value of the object once it is created , when we try to change it from the front it actually creates a new object and points it to the new memory location