import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import interpolate

def alpha_func(plane,coor1,coor2):
    '''
    plane is assumed to be already calculated
    '''
    return (plane-coor1)/(coor2-coor1)

def plane_func(index,plane1,d):
    '''
    Parameters:
    ----------
    index :: integer
      index to evaluate at, index in (1,...,n)
    
    plane1 :: float 
      location of plane with index 1 
    
    d :: float 
      distance between planes
    
    Returns:
    -------
    plane_location :: float 
      the location of the plane of the specified index
    
    '''
    return plane1 + (index-1)*d

def voxel_length(alpha,index,d12):
    '''
    Parameters:
    ----------
    alpha :: list
      list of alpha values
    
    index :: integer
      index to evaluate at, index in (1,...,n)
    
    d12 :: float
      distance from point one to point two
    
    Returns:
    -------
    voxel_length :: float
      voxel intersection length
      
    '''
    
    return d12*(alpha[index]-alpha[index-1])

def voxel_indices(plane1s,coor1s,coor2s,distances,alpha,index):
    '''
    Parameters:
    ----------
    plane1s :: array
      coordinate plane (1) in form (x,y,z)
    
    coor1s :: array
      coordinate one in form (x,y,z)
    
    coor2s :: float
      coordinate two in form (x,y,z)
    
    distances :: float
      distance between two planes in form (x,y,z)
      
    alpha :: array
      ordered set of alpha values
    
    index :: integer
      i in 1,...,nfinal
    
    Returns:
    -------
    voxel_index :: integer
      voxel index for the specific coordinate 
    
    '''
    a_mid = alpha_mid(alpha,index)
    
    i = int(np.floor(1 + (coor1s[0] + a_mid*(coor2s[0]-coor1s[0])-plane1s[0])/distances[0]))
    j = int(np.floor(1 + (coor1s[1] + a_mid*(coor2s[1]-coor1s[1])-plane1s[1])/distances[1]))
    k = int(np.floor(1 + (coor1s[2] + a_mid*(coor2s[2]-coor1s[2])-plane1s[2])/distances[2]))
    
    return (i,j,k)

def alpha_mid(alpha,index):
    '''
    Parameters:
    ----------
    alpha :: array
      ordered set of alpha values
    
    index :: integer
      i in 1,...,nfinal
    
    Returns:
    -------
    alpha_mid :: float 
    
    '''
    return (alpha[index]+alpha[index-1])/2

def plot_grid_3D(size,bins,ifig=None,colour='b'):
    '''
    size :: tuple
      (x size, y size, z size) in centimeters
    
    bins :: tuple 
      (number of x bins, number of y bins, number of z bins)
    '''
    xlines = np.linspace(0,size[0],bins[0]+1)
    ylines = np.linspace(0,size[1],bins[1]+1)
    zlines = np.linspace(0,size[2],bins[2]+1)
    
    plt.close(ifig)
    fig = plt.figure(ifig)
    ax = plt.axes(projection='3d')
    
    for z in zlines:
        for x in xlines:
            ax.plot3D([x,x],[0,size[1]],z,colour)
        for y in ylines:
            ax.plot3D([0,size[0]],[y,y],z,colour)
    
    for y in ylines:
        for x in xlines:
            ax.plot3D([x,x],[y,y],[0,size[2]],colour)
    
    return ax,fig

def Siddon(num_planes,voxel_lengths,beam_coor,ini_planes,plot=False):
    '''
    Parameters:
    ----------
    num_planes :: tuple (3)
      (Nx,Ny,Nz) for a CT array of (Nx-1,Ny-1,Nz-1) voxels
    
    voxel_lengths :: tuple (3)
      distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm
    
    beam_coor :: tuple (3,2)
      initial and final coordinates of the beam in the form ((x1,x2),(y1,y2),(z1,z2))
    
    ini_planes :: tuple (3)
      initial plane coordinates
    
    plot :: bool
      if True, plots a graphical representation of the problem
    
    Returns:
    -------
    voxel_info :: list 
      list of dictionaries each with keys 'd' (distance spent in voxel in cm), and 'indices' (the (x,y,z) coordinates of the voxel)
    
    '''
    coor_values = {'x':{},'y':{},'z':{}}

    coor_values['x']['N'] = num_planes[0]
    coor_values['y']['N'] = num_planes[1]
    coor_values['z']['N'] = num_planes[2]

    coor_values['x']['d'] = voxel_lengths[0]
    coor_values['y']['d'] = voxel_lengths[1]
    coor_values['z']['d'] = voxel_lengths[2]

    coor_values['x']['1,2'] = beam_coor[0]
    coor_values['y']['1,2'] = beam_coor[1]
    coor_values['z']['1,2'] = beam_coor[2]

    coor_values['x']['plane'] = [ini_planes[0]] # this ends up being min,max
    coor_values['y']['plane'] = [ini_planes[1]]
    coor_values['z']['plane'] = [ini_planes[2]]

    for key in coor_values.keys():
        coor_values[key]['plane'].append(plane_func(coor_values[key]['N'],coor_values[key]['plane'][0],coor_values[key]['d']))

    for key in coor_values.keys():
        if coor_values[key]['1,2'][1] - coor_values[key]['1,2'][0] != 0:
            coor_values[key]['alpha_minmax'] = (alpha_func(coor_values[key]['plane'][0],coor_values[key]['1,2'][0],coor_values[key]['1,2'][1]),alpha_func(coor_values[key]['plane'][-1],coor_values[key]['1,2'][0],coor_values[key]['1,2'][1]))
        else:
            coor_values[key]['alpha_minmax'] = (0,1) # set to this so that it doesn't affect later business

    alpha_min = max(0,min(coor_values['x']['alpha_minmax']),min(coor_values['y']['alpha_minmax']),min(coor_values['z']['alpha_minmax']))
    alpha_max = min(1,max(coor_values['x']['alpha_minmax']),max(coor_values['y']['alpha_minmax']),max(coor_values['z']['alpha_minmax']))
    
    
    for key in coor_values.keys():
        if coor_values[key]['1,2'][1] - coor_values[key]['1,2'][0] >= 0:
            indmin = coor_values[key]['N'] - (coor_values[key]['plane'][-1]-alpha_min*(coor_values[key]['1,2'][1] - coor_values[key]['1,2'][0])-coor_values[key]['1,2'][0])/coor_values[key]['d']
            indmax = 1 - (coor_values[key]['plane'][0]-alpha_max*(coor_values[key]['1,2'][1] - coor_values[key]['1,2'][0])-coor_values[key]['1,2'][0])/coor_values[key]['d']
            indmin = int(np.ceil(indmin))
            indmax = int(np.floor(indmax))
            coor_values[key]['indminmax'] = (indmin,indmax)
        else:
            indmin = coor_values[key]['N'] - (coor_values[key]['plane'][-1]-alpha_max*(coor_values[key]['1,2'][1] - coor_values[key]['1,2'][0])-coor_values[key]['1,2'][0])/coor_values[key]['d']
            indmax = 1 - (coor_values[key]['plane'][0]-alpha_min*(coor_values[key]['1,2'][1] - coor_values[key]['1,2'][0])-coor_values[key]['1,2'][0])/coor_values[key]['d']
            indmin = int(np.ceil(indmin))
            indmax = int(np.floor(indmax))
            coor_values[key]['indminmax'] = (indmin,indmax)
    
    alpha_coor_set = {}

    for key in coor_values.keys():
        if coor_values[key]['1,2'][1] - coor_values[key]['1,2'][0] > 0:
            coor_values[key]['alpha_set'] = alpha_func(plane_func(np.array([n for n in range(coor_values[key]['indminmax'][0],coor_values[key]['indminmax'][1]+1)]),coor_values[key]['plane'][0],coor_values[key]['d']),coor_values[key]['1,2'][0],coor_values[key]['1,2'][1])
        elif coor_values[key]['1,2'][1] - coor_values[key]['1,2'][0] < 0:
            coor_values[key]['alpha_set'] = alpha_func(plane_func(np.array([n for n in range(coor_values[key]['indminmax'][1],coor_values[key]['indminmax'][0]-1,-1)]),coor_values[key]['plane'][0],coor_values[key]['d']),coor_values[key]['1,2'][0],coor_values[key]['1,2'][1])
        else:
            coor_values[key]['alpha_set'] = []    
    
    alpha = [alpha_min,alpha_max] + list(coor_values['x']['alpha_set']) + list(coor_values['y']['alpha_set']) + list(coor_values['z']['alpha_set'])
    alpha = np.sort(list(alpha))
    
    
    nfinal = 1
    d12 = 0
    for key in coor_values.keys():
        if coor_values[key]['1,2'][1] - coor_values[key]['1,2'][0] != 0:
            nfinal += coor_values[key]['indminmax'][1] - coor_values[key]['indminmax'][0] + 1 
        d12 += (coor_values[key]['1,2'][1] - coor_values[key]['1,2'][0])**2
    d12 = np.sqrt(d12)
    
    voxel_info = []

    for i in range(1,nfinal+1):
        length = voxel_length(alpha,i,d12)
        indices = voxel_indices((coor_values['x']['plane'][0],coor_values['y']['plane'][0],coor_values['z']['plane'][0]),(coor_values['x']['1,2'][0],coor_values['y']['1,2'][0],coor_values['z']['1,2'][0]),(coor_values['x']['1,2'][1],coor_values['y']['1,2'][1],coor_values['z']['1,2'][1]),(coor_values['x']['d'],coor_values['y']['d'],coor_values['z']['d']),alpha,i)
        voxel_info.append({})
        voxel_info[i-1]['d'] = length
        voxel_info[i-1]['indices'] = indices
    
    if plot:
        ax,fig = plot_grid_3D((coor_values['x']['plane'][1]-coor_values['x']['plane'][0],coor_values['y']['plane'][1]-coor_values['y']['plane'][0],coor_values['z']['plane'][1]-coor_values['z']['plane'][0]),(coor_values['x']['N']-1,coor_values['y']['N']-1,coor_values['z']['N']-1),colour='g')
        ax.plot3D((coor_values['x']['1,2'][0],coor_values['x']['1,2'][1]),(coor_values['y']['1,2'][0],coor_values['y']['1,2'][1]),(coor_values['z']['1,2'][0],coor_values['z']['1,2'][1]),'r')
    
    return(voxel_info)

# filename variable is going to have to change as I move to not just calculating water 
def TERMA(num_planes,voxel_lengths,beam_coor,ini_planes,ini_energy,filename):
    '''
    Parameters:
    ----------
    num_planes :: tuple (3)
      (Nx,Ny,Nz) for a CT array of (Nx-1,Ny-1,Nz-1) voxels
    
    voxel_lengths :: tuple (3)
      distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm
    
    beam_coor :: tuple (3,2)
      initial and final coordinates of the beam in the form ((x1,x2),(y1,y2),(z1,z2))
    
    ini_planes :: tuple (3)
      initial plane coordinates
    
    ini_energy :: float 
      the initial energy of the ray in MeV
    
    filename :: str 
      name of the file that contains values for energy absorption coefficients 
    
    Returns:
    -------
    voxel_info :: list 
      list of dictionaries each with keys 'd' (distance spent in voxel in cm), 'indices' (the (x,y,z) coordinates of the voxel),
      and 'fraction_lost' (the fraction of the initial number of photons lost in that voxel)
    '''
    coeff_array = np.loadtxt(filename,skiprows=2,dtype=float)

    e_a_coeffs = [] # energy absorption coefficients

    for row in coeff_array:
        e_a_coeffs.append([row[0],row[1]])
    e_a_coeffs = np.array(e_a_coeffs)
    
    # exponentially interpolate 
    mu_energy = interpolate.interp1d(e_a_coeffs.T[0],e_a_coeffs.T[1],kind='nearest',fill_value='extrapolate')
    mu = lambda energy, material: mu_energy(energy) # CHANGE THIS LATER TO A REAL FUNCTION
    
    voxel_info = Siddon(num_planes,voxel_lengths,beam_coor,ini_planes)
    
#     energy = ini_energy
    
#     for index in range(len(voxel_info)):
#         energy_lost = energy - energy*np.exp(-mu(energy,voxel_info[index]['indices'])*voxel_info[index]['d'])
#         voxel_info[index]['energy_lost'] = energy_lost
#         energy = energy - energy_lost
    fraction = 1
    for index in range(len(voxel_info)):
        fraction_lost = fraction - fraction*np.exp(-mu(ini_energy,voxel_info[index]['indices'])*voxel_info[index]['d'])
        voxel_info[index]['fraction_lost'] = fraction_lost
        fraction = fraction - fraction_lost
        
    
    return voxel_info 
    