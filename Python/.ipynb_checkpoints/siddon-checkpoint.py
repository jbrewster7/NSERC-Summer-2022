import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import interpolate
from topas2numpy import BinnedResult
from multiprocessing import Pool
import pickle

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

def density(index):
    '''
    THIS FUNCTION IS ONLY FOR WATER RIGHT NOW 
    
    Parameters:
    ----------
    index :: tuple (3)
      indices of the voxel whose density you want
    
    Returns:
    -------
    density :: float
      density of that voxel in g/cm^3 
    
    '''
    
    return 1

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

def TERMA(num_planes,voxel_lengths,beam_coor,ini_planes,beam_energy,ini_fluence,mu_l,mu_m):
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
    
    beam_energy :: numpy array
      vector containing the the beam energies in MeV (corresponding to ini_fluence)
    
    ini_fluence :: numpy array
      vector containig the initial photon fluences in cm^-2 (corresponding to beam_energy)
    
    mu_l :: function
      function that takes energy and material as arguments and returns linear energy absorption coefficient
    
    mu_m :: function
      function that takes energy and material as arguments and returns mass energy absorption coefficient
    
    Returns:
    -------
    voxel_info :: list 
      list of dictionaries each with keys 'd' (distance spent in voxel in cm), 'indices' (the (x,y,z) indices of the voxel),
      and 'TERMA' (the total energy released per unit mass in that voxel)
    
    '''
    
    voxel_info = Siddon(num_planes,voxel_lengths,beam_coor,ini_planes)
    
    # this is photon fluence not energy fluence
    fluence = ini_fluence
    
    # DENSITY FUNCTION IS NOT REAL YET 
    # this is also calculating TERMA at the end of each voxel not the middle 
    intermediate_list = []
    for index in range(len(voxel_info)):
        if voxel_info[index]['d'] != 0:
            voxel_info[index]['TERMA'] = sum(beam_energy*fluence*mu_m(beam_energy,voxel_info[index]['indices'])/density(voxel_info[index]['indices']))
            intermediate_list.append(voxel_info[index])
            fluence = fluence*np.exp(-mu_l(beam_energy,voxel_info[index]['indices'])*voxel_info[index]['d'])

    voxel_info = intermediate_list
    
    return voxel_info
    

def Superimpose(voxel_info,voxel_array,kernel_func,center_coor,voxel_size_ratio):
    '''
    Parameters:
    ----------
    voxel_info :: list 
      list of dictionaries each with keys 'd' (distance spent in voxel in cm), 'indices' (the (x,y,z) indices of the voxel),
      and 'TERMA' (the total energy released per unit mass in that voxel)
    
    voxel_array :: numpy array 
      contains all of the (x,y,z) coordinants of all of the voxels 
    
    kernel_func :: function 
      interpolated kernel 
    
    center_coor :: tuple (3,3)
      coordinates of the centre of the kernel
    
    voxel_size_ratio :: tuple (3)
      (x,y,z) ratios of CT voxel size divided by kernel voxel size 
    
    Returns:
    -------
    energy_deposited :: numpy array
      energy deposited from that specific voxel
    
    '''
    
    energy_deposited = []
    kernel_value_total = 0
    voxel_diff = ['','','']
    
    for n in range(len(voxel_array)):
        voxel_diff[0] = voxel_array[n][0] - (voxel_info['indices'][0]-1)
        voxel_diff[1] = voxel_array[n][1] - (voxel_info['indices'][1]-1)
        voxel_diff[2] = voxel_array[n][2] - (voxel_info['indices'][2]-1)

        kernel_value = kernel_func((center_coor[0]+voxel_diff[0]*voxel_size_ratio[0],center_coor[1]+voxel_diff[1]*voxel_size_ratio[1],center_coor[2]+voxel_diff[2]*voxel_size_ratio[2]))
        energy_deposited.append(kernel_value * voxel_info['TERMA'])
        kernel_value_total += kernel_value
    
    # not 100% sure it needs to be a numpy array
    energy_deposited = np.array(energy_deposited)
    
    if kernel_value_total != 0:
        energy_deposited = energy_deposited/kernel_value_total
    
    return energy_deposited

def mask_superimpose(voxel_information):
    '''
    masks Superimpose() so that it can be called by Pool
    '''
    voxel_array = pickle.load(open('dose_calc_variables/voxel_array.pickle','rb'))
    kernel_func = pickle.load(open('dose_calc_variables/kernel_func.pickle','rb'))
    center_coor = pickle.load(open('dose_calc_variables/center_coor.pickle','rb'))
    voxel_size_ratio = pickle.load(open('dose_calc_variables/voxel_size_ratio.pickle','rb'))
    
    return Superimpose(voxel_information,voxel_array,kernel_func,center_coor,voxel_size_ratio)

def Superposition(kernel_array,kernel_size,num_planes,voxel_lengths,voxel_info,num_cores):
    '''
    Parameters:
    ----------
    kernel_array :: numpy array 
      array with normalized kernel: should be an odd number of voxels, interacting in the center
    
    kernel_size :: tuple (3)
      (x,y,z) dimensions of the kernel in cm 
    
    num_planes :: tuple (3)
      (Nx,Ny,Nz) for a CT array of (Nx-1,Ny-1,Nz-1) voxels
    
    voxel_lengths :: tuple (3)
      distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm
    
    voxel_info :: list 
      a list of a list of dictionaries each with keys 'd' (distance spent in voxel in cm), 
      'indices' (the (x,y,z) indices of the voxel), and 'TERMA' (the TERMA).
      each element of the initial list corresponds to a different ray
    
    num_cores :: integer 
      number of cores to use 
    
    Returns:
    -------
    energy_deposit :: numpy array (Nx-1,Ny-1,Nz-1)
      numpy array in the same form as one gets from taking the data ['Sum'] from a topas2numpy BinnedResult object
    
    '''
    # num_cores = 8 # really unsure if core is the right word here 
    
    Nx = num_planes[0]
    Ny = num_planes[1]
    Nz = num_planes[2]
    
    dx = voxel_lengths[0]
    dy = voxel_lengths[1]
    dz = voxel_lengths[2]
    
    kernel_info = {}
    kernel_info['x'] = {}
    kernel_info['y'] = {}
    kernel_info['z'] = {}

    kernel_info['x']['bins'] = len(kernel_array)
    kernel_info['y']['bins'] = len(kernel_array[0])
    kernel_info['z']['bins'] = len(kernel_array[0][0])

    kernel_info['x']['size'] = kernel_size[0]
    kernel_info['y']['size'] = kernel_size[1]
    kernel_info['z']['size'] = kernel_size[2]

    kernel_info['x']['voxel_size'] = kernel_info['x']['size']/kernel_info['x']['bins']
    kernel_info['y']['voxel_size'] = kernel_info['y']['size']/kernel_info['y']['bins']
    kernel_info['z']['voxel_size'] = kernel_info['z']['size']/kernel_info['z']['bins']
        
    # this x,y,z are just for kernel_func
    x = np.linspace(0,kernel_info['x']['bins']-1,kernel_info['x']['bins'])
    y = np.linspace(0,kernel_info['y']['bins']-1,kernel_info['y']['bins'])
    z = np.linspace(0,kernel_info['z']['bins']-1,kernel_info['z']['bins'])
    
    kernel_func = interpolate.RegularGridInterpolator((x,y,z),kernel_array,bounds_error=False,fill_value=0)
    
    center_coor = (int(np.floor(len(kernel_array)/2)),int(np.floor(len(kernel_array[0])/2)),int(np.floor(len(kernel_array[0][0])/2)))
        
    # making array for labelling voxels 
    x_voxels = np.linspace(0,Nx-2,Nx-1,dtype=np.uint16)
    y_voxels = np.linspace(0,Ny-2,Ny-1,dtype=np.uint16)
    z_voxels = np.linspace(0,Nz-2,Nz-1,dtype=np.uint16)
    
    # this is where I can lower size of data too 
    voxel_array = np.array([[x,y,z] for x in x_voxels for y in y_voxels for z in z_voxels])
        
    energy_deposit = []
    
    pickle.dump(voxel_array,open('dose_calc_variables/voxel_array.pickle','wb'))
    pickle.dump(kernel_func,open('dose_calc_variables/kernel_func.pickle','wb'))
    pickle.dump(center_coor,open('dose_calc_variables/center_coor.pickle','wb'))
    pickle.dump((dx/kernel_info['x']['voxel_size'],dy/kernel_info['y']['voxel_size'],dz/kernel_info['z']['voxel_size']),open('dose_calc_variables/voxel_size_ratio.pickle','wb'))
    
    p = Pool(num_cores)
    
    for ray in range(len(voxel_info)):
        energy_deposit.append([])
        
        energy_deposit[ray] = p.map(mask_superimpose,voxel_info[ray])
        
        # energy_deposit[ray] = [Superimpose(voxel_info[ray][voxel_ind],voxel_array,kernel_func,center_coor,(dx/kernel_info['x']['voxel_size'],dy/kernel_info['y']['voxel_size'],dz/kernel_info['z']['voxel_size'])) for voxel_ind in range(len(voxel_info[ray]))]
        
        energy_deposit[ray] = np.array(sum(energy_deposit[ray]))
    
    p.close()
    
    energy_deposit = np.array(sum(energy_deposit))
    energy_deposit = energy_deposit.reshape(Nx-1,Ny-1,Nz-1)
    
    return energy_deposit
    
    
def Dose_Calculator(num_planes,voxel_lengths,beam_coor,ini_planes,beam_energy,ini_fluence,filename,kernelname,kernel_size,num_cores):
    '''
    Parameters:
    ----------
    num_planes :: tuple (3)
      (Nx,Ny,Nz) for a CT array of (Nx-1,Ny-1,Nz-1) voxels
    
    voxel_lengths :: tuple (3)
      distances between the x,y,z planes (also the lengths of the sides of the voxels) in cm
    
    beam_coor :: list of tuples (3,2)
      list of initial and final coordinates of the ray in the form ((x1,x2),(y1,y2),(z1,z2)), 
      list contains one tuple for each ray 
    
    ini_planes :: tuple (3)
      initial plane coordinates
    
    beam_energy :: float 
      the energy of the beam in MeV
    
    ini_fluence :: float
      the initial photon fluence in cm^-2
    
    filename :: str 
      name of the file that contains values for energy absorption coefficients 
    
    kernelname :: str
      name or pathway of file from TOPAS that contains kernel information
    
    kernel_size :: tuple (3)
      (x,y,z) dimensions of the kernel in cm 
    
    num_cores :: integer 
      number of cores to use 
    
    Returns:
    -------
    energy_deposit :: numpy array (Nx-1,Ny-1,Nz-1)
      numpy array in the same form as one gets from taking the data ['Sum'] from a topas2numpy BinnedResult object
    
    '''
    # making mu interpolation function
    coeff_array = np.loadtxt(filename,skiprows=2,dtype=float)
    
    # exponentially interpolate 
    mu_linear = interpolate.interp1d(np.log(coeff_array.T[0]),np.log(coeff_array.T[1]),kind='linear',fill_value='extrapolate')
    mu_l = lambda energy, material: np.exp(mu_linear(np.log(energy))) # CHANGE THIS LATER TO A REAL FUNCTION

    mu_mass = interpolate.interp1d(np.log(coeff_array.T[0]),np.log(coeff_array.T[2]),kind='linear',fill_value='extrapolate')
    mu_m = lambda energy, material: np.exp(mu_mass(np.log(energy))) # CHANGE THIS LATER TO A REAL FUNCTION
    
    voxel_info = []
    
    beam_energy = np.array(beam_energy)
    ini_fluence = np.array(ini_fluence)
    
    for n in range(len(beam_coor)): 
        voxel_info.append(TERMA(num_planes,voxel_lengths,beam_coor[n],ini_planes,beam_energy,ini_fluence/len(beam_coor),mu_l,mu_m))
    
    # I don't think I really need to do this anymore... but it kinda keeps it clean so idk
    # maybe take this out later
    kernel_array_raw = BinnedResult(kernelname).data['Sum'] # non-normalized array
    kernel_array = kernel_array_raw/np.sum(kernel_array_raw) # normalized array
    
    energy_deposit = Superposition(kernel_array,kernel_size,num_planes,voxel_lengths,voxel_info,num_cores)
    return energy_deposit