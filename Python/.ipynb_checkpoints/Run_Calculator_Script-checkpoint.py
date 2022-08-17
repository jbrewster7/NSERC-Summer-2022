import dose_calculator as dc
import pickle
import numpy as np

def coorform(form,arr):
    '''
    Parameters:
    ----------
    form :: str
      what form you want to have the return array in
      (i.e. 'x', 'y', 'z' correspond to x first, ...)
    
    arr :: numpy array
      array of topas voxels
    
    Returns:
    -------
    newarr :: numpy array
    '''
    newarr = []
    if form == 'x':
        newarr = arr
    elif form == 'y':
        for j in range(len(arr[0])):
            plt_y = []
            for n in range(len(arr)):
                plt_y.append(arr[n][j])
            newarr.append(plt_y)
    elif form == 'z':
        for k in range(len(arr[0][0])):
            plt_z = []
            for j in range(len(arr[0])):
                plt_z.append([])
                for i in range(len(arr)):
                    plt_z[j].append(arr[i][j][k])
            newarr.append(plt_z)
    else:
        raise ValueError('Invalid value for form. Only allowed values are \'x\', \'y\', \'z\'')
    return np.array(newarr)

# for a CT array of (Nx-1,Ny-1,Nz-1) voxels
Nx = 229
Ny = 131
Nz = 116

# distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm
dx = 0.251
dy = 0.251
dz = 0.25

# initial and final coordinates of the beam
x1,x2 = (0,0)
y1,y2 = (0,0)
z1,z2 = (-(Nz-1)*dz/2+1.75,(Nz-1)*dz/2+1.75)

# adjustment from center 
adjust = 0.025

# initial plane coordinates (cm)
xplane1 = -(Nx-1)*dx/2 + 5.773
yplane1 = -(Ny-1)*dy/2 - 7
zplane1 = -(Nz-1)*dz/2 + 1.75

# beam info and filename
beam_energy = [0.120] # in MeV
fluence_0 = [3.183098862 * 10**8] # photon/cm^2
# fluence_0 = 2.53 * 10**8 # photon/cm^2
# fluence_0 = 9.93 * 10**8 # photon/cm^2
# filename = 'energy_absorption_coeff.txt'

# angular (in radians) and positional spread (in cm)
angle_spread_x = 0
angle_spread_y = 0
angle_spread_z = 0

pos_spread_x = 0.01
pos_spread_y = 0.01
pos_spread_z = 0

angle_spread = (angle_spread_x,angle_spread_y,angle_spread_z)
position_spread = (pos_spread_x,pos_spread_y,pos_spread_z)

# densities
myArray2 = []
fo = open("lung_patient_python/lung_densities.egsphant", "r")
for line in fo:
    for nbr in line.split():
        myArray2.append(float(nbr))
densities = np.reshape(myArray2,[228,130,115],order='f').T
densities = coorform('z',densities)

print(np.shape(densities))

# file names for energy absorption coefficients
filenames = ['energy_absorption_coeffs_air.txt','energy_absorption_coeffs.txt','energy_absorption_coeffs_lung.txt','rib_bone_coeffs.txt']

# kernel info
kernelname_air = '../Topas/AirKernel.csv'
kernelname_water = '../Topas/Kernels/WaterKernel13.csv'
kernelname_lung = '../Topas/Kernels/LungKernel2.csv'
kernelname_rib_bone = '../Topas/Kernels/BoneKernel5.csv'
kernelnames = [kernelname_air,kernelname_water,kernelname_lung,kernelname_rib_bone]
# kernelname = '../Topas/RealKernel1.csv'
kernel_size = (4,4,4) # cm 

# effective distance from center of kernel 
eff_dist = (2,2,2) # cm

# making materials array 
myArray = []
fo = open("lung_patient_python/lung_materials.egsphant", "r")
for line in fo:
    for nbr in line.split():
        for n in nbr:
            myArray.append(int(n))
            
materials = np.reshape(myArray[:],[228,130,115],order='f').T
materials = np.array(materials,dtype=int)-1
materials = coorform('z',materials)

print(np.shape(materials))

# number of cores to use
num_cores = 16

# Siddon((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),plot=True)
# terma = sd.TERMA((Nx,Ny,Nz),(dx,dy,dz),((x1,x2),(y1,y2),(z1,z2)),(xplane1,yplane1,zplane1),[beam_energy],[fluence_0],mu)

dose = dc.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1+adjust,x2+adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1+adjust,y2+adjust),(z1,z2)),((x1+adjust,x2+adjust),(y1-adjust,y2-adjust),(z1,z2)),((x1-adjust,x2-adjust),(y1-adjust,y2-adjust),(z1,z2))],(xplane1,yplane1,zplane1),beam_energy,fluence_0,angle_spread,position_spread,densities,filenames,kernelnames,kernel_size,eff_dist,materials,num_cores)
#dose = dc.Dose_Calculator((Nx,Ny,Nz),(dx,dy,dz),[((x1+adjust,x2+adjust),(y1+adjust,y2+adjust),(z1,z2))],(xplane1,yplane1,zplane1),beam_energy,fluence_0,angle_spread,position_spread,densities,filenames,kernelnames,kernel_size,eff_dist,materials,num_cores)

pickle.dump(dose,open('lungpatient_2.pickle','wb'))
