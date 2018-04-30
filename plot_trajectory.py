import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = []
y = []
z = []

with open('test_trajectory.txt') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        x.append(float(row[1]))
        y.append(float(row[2]))
        z.append(-float(row[3]))

x_real = []
y_real = []
z_real = []

with open('real_trajectory.txt') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        x_real.append(float(row[0]))
        y_real.append(float(row[1]))
        z_real.append(-float(row[2]))

#fig = plt.figure()
#ax = Axes3D(fig)

#ax.scatter(x, y, z, c='b')
#ax.scatter(x, y, z, c='r')

#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')


#plt.show()

plt.scatter(x, y, color='blue')
plt.scatter(x_real, y_real, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
