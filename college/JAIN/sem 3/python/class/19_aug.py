# Q1
'''

a = int(input("Enter the 1st int: "))
b = int(input("Enter the 2nd int: "))

c = float(input("Enter the 1st float: "))
d = float(input("Enter the 2nd float: "))

print(str(a) , str(b), str(c), str(d))

#Q2
e = "500"

print(int(e), float(e))

#Q3,
w = input("Enter the bool: ")
print(bool(w))
'''
#Q4 
a = input("Enter smtg: ")
b = input("Enter smtg: ")
try:
    con1 = int(a)
    con2 = int(b)
    print(f"Product of {a} and {b} is {con1 * con2}")

except ValueError:
    print("Couldn't convert to int and couldn't perform product opperation")

try:
    con3 = float(a)
    con4 = float(b)
    print(f"The sum of floats {a} and {b} is {con3 + con4}")
except ValueError:
    print("Couldn't convert to float and couldn't perform sum opperation")