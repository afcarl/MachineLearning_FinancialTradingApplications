# Mark Trinquero
# MC3-P1 


# GENERATING ALGO: this algorithm will generate non-linear data points for x1, x2, y.  This will ensure that KNN learner outperforms 
# the linear regression approach due to the non-linear nature of the data set.  Additional noise is also added to the data to add
# another layer of randomization amoung points.


import csv
from random import randint






f = open('best4KNN.csv', 'w')

writer = csv.writer(f)
counter = 1
data_size = 1001

while counter < data_size:

    x1 = float(randint(0, 15))
    x2 = float(randint(0, 15))
    add_noise = float(randint(0, 10))
    y = float((x1**2) + (x2**3) + add_noise)
    writer.writerow( [ x1 , x2 , y ] )
    counter = counter + 1

f.close()
