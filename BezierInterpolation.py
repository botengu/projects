# import excel 
import csv
import numpy as np
from numpy import sin, cos, pi  
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Lin
from OCC.Core.Geom import Geom_Line
from OCC.Core.GeomAPI import GeomAPI_IntCS
from OCC.Core.Geom import Geom_BezierSurface
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomLProp import GeomLProp_SurfaceTool

def f(x, y):
    return ((y*math.pi/180)**2+(1-x)**2)

def plane(x,y,t): 
    return t*np.ones([len(y),len(x)],dtype=float)

def heightOnSurf(h_surf, x,y):
    pnt, vec = gp_Pnt(x,y,0), gp_Dir(0, 0, 1)
    ray = Geom_Line(gp_Lin (pnt, vec))
    if GeomAPI_IntCS(ray, h_surf).NbPoints() >0:
        pt = GeomAPI_IntCS(ray, h_surf).Point(1)
        return pt.Z()
    else:
        return 0 

# Open the densities 
qual = []

with open('C:\\Users\\AdminSi\\Desktop\\CAGD\\PythonOcc\\BezierSurface\\Manufacturability.csv') as csvfile:
    readCSV = csv.reader(csvfile)
    for row in readCSV:
        for i in row:
            qual.append(float(i)) 

angles = 11 
radii = 13
qualities = np.reshape(qual, (angles, radii))


array_surface = TColgp_Array2OfPnt(1, angles,1, radii)
s = 1
for i in range(angles):
    for j in range(radii):
        array_surface.SetValue(i+1,j+1, gp_Pnt((70-5*i),s*(0.2+0.05*j), qualities[i,j] ))

Bz_surf = Geom_BezierSurface(array_surface) 

x = np.arange(0.2, 0.85, 0.05)
y = np.arange(70, 15, -5)
X, Y = np.meshgrid(x, y)

Z = np.ones((angles, radii),dtype =float)

for i in range(angles):
    for j in range(radii):
        Z[i,j]= heightOnSurf(Bz_surf,  float(70-5*i),float(s*(0.2+0.05*j))) 


Zplane = plane(x,y,29)


fig = plt.figure()
ax = plt.axes(projection='3d')


#--- PLOT surfaces ----------------------------- 


# Theoretical 
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha = 1, edgecolor='none', zorder =1)
#ax.plot_surface(X, Y, Zplane,color = 'green', alpha =0.67, zorder = -1)
#ax.contour(X, Y, Z-Zplane, lw=1, levels = [0])
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(30,-20)
ax.set_xlabel("Minimum Feature Size (mm)")
ax.set_ylabel("Overhanging Angles (°)")
ax.set_zlabel("Micro Surface Roughness")
plt.show()
