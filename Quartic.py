# import excel 
import csv
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import math
from numpy import sin, cos, pi  
from mpl_toolkits import mplot3d
from matplotlib import cm
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Lin
from OCC.Core.Geom import Geom_Line
from OCC.Core.GeomAPI import GeomAPI_IntCS
from OCC.Core.Geom import Geom_BezierSurface
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomLProp import GeomLProp_SurfaceTool


# --------------------- FUNCTIONS --------------------
def plane(x,y,t): 
    return t*np.ones([len(y),len(x)],dtype=float)


#P = np.random.rand(3,10) # given point set

def quartic(x,y, a, b, c, d, e, f,g,h,i,j,k,l,m,n,o): 
  #fit quadratic surface
  return a*x**4 + b*y**4 + c*x**3*y + d*x**2*y**2 + e*x*y**3 + f*x**3 + g*y**3 + h*x*y**2 + i*x**2*y +j*x**2 + k*y**2 + l*x*y+m*x+n*y+o 

def residual(params, points):
  #total residual
  residuals = [
    p[2] - quartic(p[0], p[1],
    params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10],
    params[11], params[12],params[13],params[14]) for p in points]

  return np.linalg.norm(residuals)

# --------------------- DATA --------------------
# Open the densities 
qual = []

with open('C:\\Users\\AdminSi\\Desktop\\CAGD\\PythonOcc\\BezierSurface\\Manufacturability.csv') as csvfile:
    readCSV = csv.reader(csvfile)
    for row in readCSV:
        for i in row:
            qual.append(float(i)) 



x = np.linspace(0, 90, 10)
y = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y)

angles = 11 
radii = 13
qualities = np.reshape(qual, (angles, radii))


array_surface = []
s = 1
for i in range(angles):
    for j in range(radii):
        array_surface.append([70-5*i,0.2+0.05*j,qualities[i,j]])
P = np.array(array_surface).T
# --------------- OPTIMIZATION -------------
result = scipy.optimize.minimize(residual, 
                                 (1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1),#starting point
                                 args=P)

zvalues = quartic(X,Y, result.x[0],result.x[1],result.x[2],result.x[3],result.x[4],result.x[5],result.x[6],result.x[7],result.x[8],result.x[9]
,result.x[10],result.x[11],result.x[12],result.x[13],result.x[14])


Zplane1 = plane(x,y,29)



#--- PLOT surfaces ----------------------------- 
fig = plt.figure(figsize = (10, 7)) 
ax = plt.axes(projection ="3d") 
 
# Creating plot 


# Theoretical 
surf = ax.plot_surface(X, Y, zvalues, rstride=1, cstride=1, cmap=cm.coolwarm, alpha = 1, edgecolor='none', zorder =1)
#ax.plot_surface(X, Y, Zplane,color = 'green', alpha =0.67, zorder = -1)
#ax.contour(X, Y, Z-Zplane, lw=1, levels = [0])
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(30,-20)
ax.set_xlabel("Overhanging Angles (°)")
ax.set_ylabel("Minimum Feature Size (mm)")
ax.set_zlabel("Micro Surface Roughness")
plt.show()
''' 
 

contours1= measure.find_contours(zvalues, 0.12) 
#contours2= measure.find_contours(volnew, 0.18) 
#contours3= measure.find_contours(volnew, 0.25)
#contours4= measure.find_contours(volnew, 0.3)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(volnew, cmap=plt.cm.gray)
  
for n, contour in enumerate(contours1):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

for n, contour in enumerate(contours2):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

for n, contour in enumerate(contours3):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])

plt.show()
'''