# Mark Trinquero
# MC3-P1 

# GENERATING ALGO: this algorithm will generate randomly spaced points along a line.  This is achieved by setting x1 and x2 equal
# to the square of the index/counter.  The y value is the calculated as Y = X1 + 2X2.  These data points are then written into a
# csv file for further analysis.


import csv







# open file and write geneated data into it
f = open('best4linreg.csv', 'w')

writer = csv.writer(f)
counter = 1
data_size = 1001

while counter < data_size:

    x1 = float(counter**2)
    x2 = float(counter**2)
    y = float(x1 + (2*x2))
    writer.writerow( [ x1 , x2 , y ] )
    counter = counter + 1

f.close()
