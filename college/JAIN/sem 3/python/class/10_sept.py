'''
Q1) Nested conditions
To get input from 10 inputs using for loop inside the for loop check the first condition...if num <= 10 and num >= 1 
then make an inner if statement to check the number is odd number...If the number is odd...print the number value along with statement "the given number is odd"
else print number and the statement "the number is even". In the outer else part print the number value and display "num not in range of 1 - 10"
'''

'''for i in range(1,11):
    num = int(input("Enter a number between 1 and 10: "))
    if(num <= 10) and (num >= 1):
        if (num % 2 == 1):
            print(f"{num} - the number is odd")
        else:
            print(f"{num} - the number is even")
    else: 
        print("the given number not in range of 1 - 10")'''


'''
Looping Statement:

- For,while,nested(for) loop
- we are using range function(pre defined) used in for loop to describe the starting value and one value after the ending value will
be given as range numbers...for example for i in range(1,6): print (i)...this will print 1-5

While loop:
- onsists of initialisation of a number at the starting of a program
- inside the while loop condition 
- in the body of while loop increment or decrement operator 
'''

'''
Q2) Write a program using for loop and while loop to display only the odd numebers between the range of 1 - 5000
'''
'''
for i in range (1, 5001):
    if (i % 2 == 1):
        print(i)

while i <= 5000:
    if (i % 2 == 1):
        print(i)
        i += 1'''

'''
Nested for loop:
- consists of 1 for loop inside another the first for loop is said to be outer for loop...the 2nd one is said to be inner for loop 
- everytime the inner for loop is controlled by the outer for loop
-eg
for i in range of(1,4):
    for j in range(1,4):
        print(f"i = {i}, j = {j}, end = " "")
        print()
'''

'''
Q3) Write a simple program to perform matrix addition for 2 x 2 matrix using nested for loop 
similarly try for 3 x 3 matrix...matrix multiplication and store the result in a resultant matrix and display the result as output

'''

a = [[0]*2 for _ in range(2)]
b = [[0]*2 for _ in range(2)]
c = [[0]*2 for _ in range(2)]

for i in range(0,2):
    for j in range(0,2):
        a[i][j] = int(input(f"Enter the {i,j} value of matrix "))

for i in range(0,2):
    for j in range(0,2):
        b[i][j] = int(input(f"Enter the {i,j} value of matrix "))

for i in range(0,2):
    for j in range(0,2):
        c[i][j] = a[i][j] + b[i][j]

for i in range(0,2):
    for j in range(0,2):
        print(c[i][j])


