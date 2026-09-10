'''
Get 5 different marks as input from user for 10 different students…the marks are declared in the form of m1, m2, m3, m4, m5, find total for those marks…after finding total…find the average for the total. After finding the average check the following conditions:
If the average is <= 100 and >= 95, then print the student got A+ grade, 
If the average is <= 95 and >= 90, then print the student got A grade,
If the average is <= 90 and >= 81, then print the student got B grade,
If the average is <= 80 and >= 50, then print the student got C grade,
If the average is < 50, then print the student got backlog
Else invalid input
'''

for i in range(1,11):
    m1 = int(input(f"Enter mark 1 for student {i} : "))
    m2 = int(input(f"Enter mark 2 for student {i} : "))
    m3 = int(input(f"Enter mark 3 for student {i} : "))
    m4 = int(input(f"Enter mark 4 for student {i} : "))
    m5 = int(input(f"Enter mark 5 for student {i} : "))

    total = m1 + m2 + m3 + m4 + m5
    avg = total / 5
    if(avg <= 100 and avg >= 95):
        print(f"Student {i} got A+")
    elif(avg < 95 and avg >= 90):
        print(f"Student {i} got A")
    elif(avg < 90 and avg >= 81):
        print(f"Student {i} got B")
    elif(avg < 81 and avg >= 50):
        print(f"Student {i} got C")
    elif(avg < 50):
        print(f"Student {i} got Backlog")
    else:
      print("Invalid input")
