import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import interpolate
from topas2numpy import BinnedResult
from multiprocessing import Pool
import pickle
from numpy import linalg
from os.path import exists

def alpha_func(plane,coor1,coor2):
    '''
    Internal function to be called by Siddon().
    '''
    return (plane-coor1)/(coor2-coor1)

def plane_func(index,plane1,d):
    '''
    Internal function to be called by Siddon().
    
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
    Internal function to be called by Siddon().
    
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
    Internal function to be called by Siddon().
    
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
    Internal function to be called by Siddon().
    
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

def Siddon(num_planes,voxel_lengths,beam_coor,ini_planes):
    '''
    Ray tracing algorithm. 
    
    Source: Siddon, R. L. (1985). Fast calculation of the exact radiological path for a three dimensional CT array. In Medical Physics (Vol. 12, Issue 2). https://doi.org/10.1118/1.595715</div>
    
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
      if True, plots a graphical representation of the problem (only works for ini_planes all 0)
    
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

    coor_values['x']['plane'] = [ini_planes[0]] # this ends up being min,max planes
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
    
    return(voxel_info)

def TERMA(num_planes,voxel_lengths,beam_coor,ini_planes,angle_spread,position_spread,beam_energy,ini_fluence,mu,percent_cutoff=1,mat_array=None,air_index=None):
    '''
    Function to calculate the Total Energy Released in MAtter (TERMA) (total energy released per unit mass). 
    
    Can be called externally if you are careful about parameters. Some are not intuitive.
    
    Mostly just used in calls internally from Dose_Calculator().
    
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
    
    angle_spread :: tuple (3) 
      angle of beam in (x,y,z) in radians
    
    position_spread :: tuple (3)
      position spread of beam in (x,y,z) in cm
    
    beam_energy :: numpy array
      vector containing the the beam energies in MeV (corresponding to ini_fluence)
    
    ini_fluence :: numpy array
      vector containing the initial photon fluences in cm^-2 (corresponding to beam_energy)
    
    mu :: function
      function that takes energy and material as arguments and returns either mass or linear energy absorption coefficient,
      depending on specified parameters
    
    Optional Parameters:
    -------------------
    percent_cutoff :: float 
      what percent of initial fluence to stop tracking particles 
      Default: 1
    
    mat_array :: array
      array of materials, only given if needing to neglect air.
      this parameter is only used internally: do not provide this if just using this function to calculate TERMA 
      must be provided if 'air_index' is provided
      Default: None
    
    air_index :: int
      value of air in mat_array
      must be provided if 'mat_array' is provided
      Default: None
    
    Returns:
    -------
    voxel_info :: list 
      list of dictionaries each with keys 'd' (distance spent in voxel in cm), 'indices' (the (x,y,z) indices of the voxel),
      and 'TERMA' (the total energy released per unit mass in that voxel) in MeV/g
    
    '''    
    eps = 1.e-7
    
    voxel_info = Siddon(num_planes,voxel_lengths,beam_coor,ini_planes)
    
    # this is photon fluence not energy fluence
    fluence = ini_fluence
    
    # this is also calculating TERMA at the beginning of each voxel not the middle 
    intermediate_list = []
    
    if angle_spread[0] <= eps and angle_spread[1] <= eps and angle_spread[2] <= eps:
        for index in range(len(voxel_info)):
            if voxel_info[index]['d'] != 0 and (air_index==None or mat_array[voxel_info[index]['indices'][0]-1][voxel_info[index]['indices'][1]-1][voxel_info[index]['indices'][2]-1]!=air_index) and sum(fluence)>=sum(ini_fluence)*percent_cutoff/100:
                voxel_info[index]['TERMA'] = sum(beam_energy*fluence*mu(beam_energy,voxel_info[index]['indices'],'m'))
                intermediate_list.append(voxel_info[index])
                fluence = fluence*np.exp(-mu(beam_energy,voxel_info[index]['indices'],'l')*voxel_info[index]['d'])
        voxel_info = intermediate_list
        
    
    # this is for the one fan beam case... 
    elif angle_spread[1] > eps:
        total_dist = position_spread[1]/angle_spread[1]
        for index in range(len(voxel_info)):
            if voxel_info[index]['d'] != 0 and (air_index==None or mat_array[voxel_info[index]['indices'][0]-1][voxel_info[index]['indices'][1]-1][voxel_info[index]['indices'][2]-1]!=air_index) and np.sum(fluence)>=np.sum(ini_fluence)*percent_cutoff/100:
                voxel_info[index]['TERMA'] = sum(beam_energy*fluence*mu(beam_energy,voxel_info[index]['indices'],'m')/total_dist)
                intermediate_list.append(voxel_info[index])
                total_dist += voxel_info[index]['d']
                fluence = fluence*np.exp(-mu(beam_energy,voxel_info[index]['indices'],'l')*voxel_info[index]['d'])
        voxel_info = intermediate_list
    
    return voxel_info
    
def Superimpose(indices,TERMA,energy_deposition_arrays,center_coor,mat_array):
    '''
    Internal function used in calls from Dose_Calculator().
    
    Parameters:
    ----------
    indices :: tuple (3) 
      (x,y,z) indices for that voxel (with list indexing starting at 1)
    
    TERMA :: float
      the TERMA for that voxel
    
    energy_deposition_arrays :: list of numpy arrays 
      list of energy_deposition arrays in same order as defined in Dose_Calculator()
    
    center_coor :: tuple (3)
      coordinates of the center of the energy_deposit array
    
    mat_array :: numpy array 
      array in the shape of (Nx-1,Ny-1,Nz-1) giving the material type in that voxel ('w' for water, 'l' for lung, 'b' for bone)
      WARNING: if mat_array is the wrong shape it might not give an error but it will mess with results
    
    Returns:
    -------
    energy_deposited :: numpy array
      energy deposited from that specific voxel
    
    '''
    mat_ind = mat_array[indices[0]-1][indices[1]-1][indices[2]-1]
    
    if indices[0]-1 < center_coor[0]:
        energy_deposited = energy_deposition_arrays[mat_ind][center_coor[0]-(indices[0]-1):]
    elif indices[0]-1 > center_coor[0]:
        energy_deposited = np.append(np.zeros(((indices[0]-1)-center_coor[0],len(energy_deposition_arrays[mat_ind][0]),len(energy_deposition_arrays[mat_ind][0][0]))),energy_deposition_arrays[mat_ind],axis=0)
    else:
        energy_deposited = energy_deposition_arrays[mat_ind]
    
    if indices[1]-1 < center_coor[1]:
        energy_deposited = np.delete(energy_deposited,np.arange(center_coor[1]-(indices[1]-1)),axis=1)
    elif indices[1]-1 > center_coor[1]:
        energy_deposited = np.append(np.zeros((len(energy_deposited),(indices[1]-1)-center_coor[1],len(energy_deposited[0][0]))),energy_deposited,axis=1)
    
    if indices[2]-1 < center_coor[2]:
        energy_deposited = np.delete(energy_deposited,np.arange(center_coor[2]-(indices[2]-1)),axis=2)
    elif indices[2]-1 > center_coor[2]:
        energy_deposited = np.append(np.zeros((len(energy_deposited),len(energy_deposited[0]),(indices[2]-1)-center_coor[2])),energy_deposited,axis=2)
        
    if (len(mat_array)-indices[0]) < center_coor[0]:
        energy_deposited = energy_deposited[:-(center_coor[0]-(len(mat_array)-indices[0]))]
    elif (len(mat_array)-indices[0]) > center_coor[0]:
        energy_deposited = np.append(energy_deposited,np.zeros(((len(mat_array)-indices[0]) - center_coor[0],len(energy_deposited[0]),len(energy_deposited[0][0]))),axis=0)
        
    if (len(mat_array[0])-indices[1]) < center_coor[1]:
        energy_deposited = np.delete(energy_deposited,np.arange(len(energy_deposited[0])-(center_coor[1]-(len(mat_array[0])-indices[1])),len(energy_deposited[0])),axis=1)
    elif (len(mat_array[0])-indices[1]) > center_coor[1]:
        energy_deposited = np.append(energy_deposited,np.zeros((len(energy_deposited),(len(mat_array[0])-indices[1]) - center_coor[1],len(energy_deposited[0][0]))),axis=1)
        
    if (len(mat_array[0][0])-indices[2]) < center_coor[2]:
        energy_deposited = np.delete(energy_deposited,np.arange(len(energy_deposited[0][0])-(center_coor[2]-(len(mat_array[0][0])-indices[2])),len(energy_deposited[0][0])),axis=2)
    elif (len(mat_array[0][0])-indices[2]) > center_coor[2]:
        energy_deposited = np.append(energy_deposited,np.zeros((len(energy_deposited),len(energy_deposited[0]),(len(mat_array[0][0])-indices[2]) - center_coor[2])),axis=2)
    
    energy_deposited = np.array(energy_deposited)
    energy_deposited = energy_deposited * TERMA
    
    return energy_deposited
    
def Superposition(kernel_arrays,kernel_size,num_planes,voxel_lengths,voxel_info,beam_coor,eff_distance,mat_array,num_cores):
    '''
    Internal function called by Dose_Calculator().
    
    Parameters:
    ----------
    kernel_arrays :: list of numpy arrays
      list of arrays with normalized kernels: should be an odd number of voxels, interacting in the center
      in order of [water,lung,bone]
    
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
    
    eff_distance :: tuple (3)
      how far away from center in (x,y,z) does kernel have an effect (in cm)
    
    mat_array :: numpy array 
      array in the shape of (Nx-1,Ny-1,Nz-1) giving the material type in that voxel ('w' for water, 'l' for lung, 'b' for bone)
    
    num_cores :: integer 
      number of cores to use 
    
    Returns:
    -------
    energy_deposit :: numpy array (Nx-1,Ny-1,Nz-1)
      numpy array in the same form as one gets from taking the data ['Sum'] from a topas2numpy BinnedResult object
    
    '''
    
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

    kernel_info['x']['bins'] = len(kernel_arrays[0])
    kernel_info['y']['bins'] = len(kernel_arrays[0][0])
    kernel_info['z']['bins'] = len(kernel_arrays[0][0][0])

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
    
    eff_voxels = (eff_distance[0]/dx,eff_distance[1]/dy,eff_distance[2]/dz)
    
    kernel_funcs = []
    for kernel_array in kernel_arrays:
        kernel_funcs.append(interpolate.RegularGridInterpolator((x,y,z),kernel_array,bounds_error=False,fill_value=0))
        
    center_coor = (int(np.floor(len(kernel_arrays[0])/2)),int(np.floor(len(kernel_arrays[0][0])/2)),int(np.floor(len(kernel_arrays[0][0][0])/2)))
    
    # making array for labelling voxels 
    x_voxels = np.linspace(0,Nx-2,Nx-1,dtype=np.uint16)
    y_voxels = np.linspace(0,Ny-2,Ny-1,dtype=np.uint16)
    z_voxels = np.linspace(0,Nz-2,Nz-1,dtype=np.uint16)
    
    voxel_array = np.array([[x,y,z] for x in x_voxels for y in y_voxels for z in z_voxels])
    
    energy_deposit = []
    
    CT_basis = np.array([[dx,0,0],[0,dy,0],[0,0,-dz]])
    pre_rotated_ker = np.array([[kernel_info['x']['voxel_size'],0,0],[0,kernel_info['y']['voxel_size'],0],[0,0,kernel_info['z']['voxel_size']]])
        
    for ray in range(len(voxel_info)):
        energy_deposit.append([])
        delta_x = beam_coor[ray][0][1]-beam_coor[ray][0][0]
        delta_y = beam_coor[ray][1][1]-beam_coor[ray][1][0]
        delta_z = beam_coor[ray][2][1]-beam_coor[ray][2][0]
        
        # this condition checks if it needs to remake the energy deposition arrays because the ray is at a different angle to the previous
        if ray == 0 or delta_x!=beam_coor[ray-1][0][1]-beam_coor[ray-1][0][0] or delta_y!=beam_coor[ray-1][1][1]-beam_coor[ray-1][1][0] or delta_z!=beam_coor[ray-1][2][1]-beam_coor[ray-1][2][0]:
            if delta_x == 0 and delta_y == 0 and delta_z == 0:
                raise ValueError('Rays cannot have magnitude of 0. The problem might be that ray never enters the array.')
            elif delta_x == 0 and delta_y == 0:
                kernel_basis = pre_rotated_ker
            elif delta_z == 0 and delta_x == 0:
                kernel_basis = np.array([pre_rotated_ker[0],-pre_rotated_ker[2],pre_rotated_ker[1]])
            elif delta_z == 0 and delta_y == 0:
                kernel_basis = np.array([-pre_rotated_ker[2],pre_rotated_ker[1],pre_rotated_ker[0]])
            else:
                Rx = np.array([[1,0,0],[0,delta_z/np.sqrt(delta_y**2+delta_z**2),-delta_y/np.sqrt(delta_y**2+delta_z**2)],[0,delta_y/np.sqrt(delta_y**2+delta_z**2),delta_z/np.sqrt(delta_y**2+delta_z**2)]])
                Ry = np.array([[delta_x/np.sqrt(delta_x**2+delta_z**2),0,delta_z/np.sqrt(delta_x**2+delta_z**2)],[0,1,0],[-delta_z/np.sqrt(delta_x**2+delta_z**2),0,delta_x/np.sqrt(delta_x**2+delta_z**2)]])
                Rz = np.array([[delta_x/np.sqrt(delta_x**2+delta_y**2),-delta_y/np.sqrt(delta_x**2+delta_y**2),0],[delta_y/np.sqrt(delta_x**2+delta_y**2),delta_x/np.sqrt(delta_x**2+delta_y**2),0],[0,0,1]])
                kernel_basis = np.array([Rx.dot(Ry.dot(Rz.dot(pre_rotated_ker[0]))),Rx.dot(Ry.dot(Rz.dot(pre_rotated_ker[1]))),Rx.dot(Ry.dot(Rz.dot(pre_rotated_ker[2])))])

            kernel_coors_mat = []
            for vec in CT_basis:
                kernel_coors_mat.append(linalg.solve(kernel_basis.T,vec))
            kernel_coors_mat = np.array(kernel_coors_mat).T

            num_voxel_in_eff_dist = [2*eff_distance[0]//dx,2*eff_distance[1]//dy,2*eff_distance[2]//dz]

            if num_voxel_in_eff_dist[0]%2==0:
                num_voxel_in_eff_dist[0] = num_voxel_in_eff_dist[0]+1
            if num_voxel_in_eff_dist[1]%2==0:
                num_voxel_in_eff_dist[1] = num_voxel_in_eff_dist[1]+1
            if num_voxel_in_eff_dist[2]%2==0:
                num_voxel_in_eff_dist[2] = num_voxel_in_eff_dist[2]+1

            if num_voxel_in_eff_dist[0] >= Nx:
                num_voxel_in_eff_dist[0] = (Nx%2-1)
            if num_voxel_in_eff_dist[1] >= Ny:
                num_voxel_in_eff_dist[1] = (Ny%2-1)
            if num_voxel_in_eff_dist[2] >= Nz:
                num_voxel_in_eff_dist[2] = (Nz%2-1)

            voxel_array_for_total = np.array([[x,y,z] for x in x_voxels[:int(num_voxel_in_eff_dist[0])] for y in y_voxels[:int(num_voxel_in_eff_dist[1])] for z in z_voxels[:int(num_voxel_in_eff_dist[2])]])
            
            kernel_total_values = list(np.zeros(len(kernel_arrays)))

            voxel_diff = [0,0,0]
            energy_depositions = []
            for k in kernel_total_values:
                energy_depositions.append([])

            for n in range(len(voxel_array_for_total)):
                voxel_diff[0] = voxel_array_for_total[n][0] - (num_voxel_in_eff_dist[0]//2)
                voxel_diff[1] = voxel_array_for_total[n][1] - (num_voxel_in_eff_dist[1]//2)
                voxel_diff[2] = voxel_array_for_total[n][2] - (num_voxel_in_eff_dist[2]//2)

                kernel_diff = kernel_coors_mat.dot(voxel_diff)
                
                for kernel_func_ind in range(len(kernel_funcs)):
                    kernel_value = kernel_funcs[kernel_func_ind]((center_coor[0]+kernel_diff[0],center_coor[1]+kernel_diff[1],center_coor[2]+kernel_diff[2]))

                    kernel_total_values[kernel_func_ind] += kernel_value

                    energy_depositions[kernel_func_ind].append(kernel_value)

            # the center coordinates of the energy deposition arrays 
            center_coor_en = (int(num_voxel_in_eff_dist[0]//2),int(num_voxel_in_eff_dist[1]//2),int(num_voxel_in_eff_dist[2]//2))
            
            for index in range(len(energy_depositions)):
                energy_depositions[index] = np.array(energy_depositions[index])/kernel_total_values[index]

                energy_depositions[index] = energy_depositions[index].reshape(int(num_voxel_in_eff_dist[0]),int(num_voxel_in_eff_dist[1]),int(num_voxel_in_eff_dist[2]))
        
        energy_deposit[ray] = [Superimpose(voxel_info[ray][voxel_ind]['indices'],voxel_info[ray][voxel_ind]['TERMA'],energy_depositions,center_coor_en,mat_array) for voxel_ind in range(len(voxel_info[ray]))]
        
        energy_deposit[ray] = np.array(sum(energy_deposit[ray]))
        # print('Ray #',ray,'complete.')
    
    energy_deposit = np.array(sum(energy_deposit))
    energy_deposit = energy_deposit.reshape(Nx-1,Ny-1,Nz-1)
    
    return energy_deposit
    
def Dose_Calculator(num_voxels,voxel_lengths,beam_coor,ini_planes,beam_energy,ini_fluence,angle_spread,position_spread,densities,filenames,kernelnames,kernel_size,eff_distance,mat_array,num_cores=1,coeff_units='cm^2/g',percent_cutoff=1,air_index=None):
    '''
    Function to be called by user. 
    
    Parameters:
    ----------
    num_voxels :: tuple (3)
      number of voxels in order of (x,y,z) 
    
    voxel_lengths :: tuple (3)
      lengths of the sides of the (x,y,z) voxels in cm
    
    beam_coor :: list of tuples (3,2)
      list of initial and final coordinates of the ray in the form ((x1,x2),(y1,y2),(z1,z2)), 
      list contains one tuple for each ray 
      NOTE: If ray falls exactly on boundary of two voxels, will deposit in one of the sides, not both. 
    
    ini_planes :: tuple (3)
      location of edge of (x,y,z) voxel arrays in cm
      examples:
         (-2.5,-2.5,-2.5) for a 5x5x5 cm phantom centered at (0,0,0)
         (0,0,0) for a 2x2x2 cm phantom centered at (1,1,1)
         (-3,-2,1) for a for 3x2x1 cm phantom centered at (-1.5,-1,1.5)
    
    beam_energy :: float 
      the energy of the beam in MeV
    
    ini_fluence :: float
      the initial photon fluence in cm^-2
      this is given by number of photons per cross-section area of the beam
      example:
         2.5*10**7 for a beam with 10^8 photons with a cross-sectional area of 4cm^2
    
    angle_spread :: tuple (3)
      angle of beam spread in (x,y,z) in radians
      WARNING: This parameter does not work properly and should be set to (0,0,0) unless actually working on code. 
               It **might** work for a beam moving from +z to -z with anglular spread in y but requires more testing still
    
    position_spread :: tuple (3)
      position spread of beam in (x,y,z) in cm
      NOTE: this parameter only matters if you are using angle_spread too and will not influence results for pencil beams
    
    densities :: list of float or numpy array
      list of densities of materials either in 1D list in same order as filenames and kernelnames parameters,
      or numpy array in same shape as mat_array
    
    filenames :: list of str 
      list of names of the files that contain values for energy absorption coefficients in same order as densities and kernelnames parameters
      see Extra/energy_absorption_coeffs.txt as an example of formatting 
    
    kernelnames :: list of str
      list of name or pathway of file from TOPAS that contains kernel information in same order as densities and filenames parameters
      NOTE: If you update the contents of a kernel file without renaming, it will not run with the updates unless you remove the file with the kernel name + .npy
    
    kernel_size :: tuple (3)
      (x,y,z) dimensions of the kernels in cm 
      NOTE: kernels must have the same dimensions and number of bins in each direction
    
    beam_coor :: list of tuples (3,2)
      list of initial and final coordinates of the ray in the form ((x1,x2),(y1,y2),(z1,z2)), 
      list contains one tuple for each ray 
    
    eff_distance :: tuple (3)
      how far away from center in (x,y,z) does kernel have an effect (in cm)
      must be less than kernel_size
      the smaller this is, the faster the simulation will be to execute
    
    mat_array :: numpy array of integers
      array in the shape of (number of x voxels,number of y voxels,number of z voxels) giving the material type in that voxel by index of place of material in densities, kernelnames, and filenames, with indexing starting at 0 
      NOTE: DTYPE MUST BE INTEGERS NOT FLOATS
    
    Optional Parameters:
    -------------------
    num_cores :: integer 
      number of cores to use
      NOTE: THIS PARAMETER DOES NOTHING
      Default: 1
    
    coeff_units :: str
      units of the coefficients in 'filenames', either 'cm^2/g' or 'cm^-1'
      Default: 'cm^2/g'
    
    percent_cutoff :: float 
      what percent of initial fluence to stop tracking particles 
      Default: 1
    
    air_index :: integer 
      index of air in 'filenames','densities','kernelnames' if you want air to be neglected in calculations; significantly speeds up algorithm 
      Default: None
    
    Returns:
    -------
    energy_deposit :: numpy array (Nx-1,Ny-1,Nz-1)
      numpy array in the same form as one gets from taking the data ['Sum'] from a topas2numpy BinnedResult object
    
    '''
    num_planes = [num_voxels[0]+1,num_voxels[1]+1,num_voxels[2]+1]
    
    if mat_array.dtype not in (int,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64):
        raise ValueError('dtype of \'mat_array\' must be type of integer not float')
    
    if len(kernelnames) != len(filenames):
        raise ValueError('\'filenames\' and \'kernelnames\' parameters must have the same number of items')
    
    if np.shape(np.array(densities)) == np.shape(np.array(mat_array)):
        den_array = True
    elif np.shape(np.array(densities)) == np.shape(np.array(filenames)):
        den_array = False
    else:
        raise ValueError('\'densities\' parameter must be the same shape as \'mat_array\' or \'filenames\' and \'kernelnames\'')
    
    if coeff_units == 'cm^2/g':
        density_units = True
    elif coeff_units == 'cm^-1':
        density_units = False
    else:
        raise ValueError('\'coeff_units\' parameter must be either \'cm^2/g\' or \'cm^-1\'')
    
    mu_linear = []
    mu_mass = []
    for material_name in filenames:
        coeff_array = np.loadtxt(material_name,skiprows=2,dtype=float)
        mu_linear.append(interpolate.interp1d(np.log(coeff_array.T[0]),np.log(coeff_array.T[1]),kind='linear',fill_value='extrapolate'))
        mu_mass.append(interpolate.interp1d(np.log(coeff_array.T[0]),np.log(coeff_array.T[2]),kind='linear',fill_value='extrapolate'))
        
    def mu(energy,voxel_index,kind):
        '''
        This is currently only set-up for water, lung, and cortical bone, will update later.

        Parameters:
        ----------
        energy :: numpy array or float
          energy of the beam 

        voxel_index :: tuple (3)
          (x,y,z) coordinates of the voxel 

        kind :: str 
          type of absorption coefficient ('l' for linear and 'm' for mass-energy)

        Returns:
        -------
        absorption_coefficient :: float 
          linear energy absorption coefficient in cm^{-1} or 
          mass energy absorption coefficient in cm^2/g

        '''
        # exponential interpolation
        if kind == 'l':
            if density_units:
                if den_array:
                    return np.exp(mu_linear[mat_array[voxel_index[0]-1][voxel_index[1]-1][voxel_index[2]-1]](np.log(energy)))*densities[voxel_index[0]-1][voxel_index[1]-1][voxel_index[2]-1]
                else:
                    return np.exp(mu_linear[mat_array[voxel_index[0]-1][voxel_index[1]-1][voxel_index[2]-1]](np.log(energy)))*densities[mat_array[voxel_index[0]-1][voxel_index[1]-1][voxel_index[2]-1]]
            else:
                return np.exp(mu_linear[mat_array[voxel_index[0]-1][voxel_index[1]-1][voxel_index[2]-1]](np.log(energy)))
        elif kind == 'm':
            if density_units:
                return np.exp(mu_mass[mat_array[voxel_index[0]-1][voxel_index[1]-1][voxel_index[2]-1]](np.log(energy)))
            else:
                if den_array:
                    return np.exp(mu_mass[mat_array[voxel_index[0]-1][voxel_index[1]-1][voxel_index[2]-1]](np.log(energy)))/densities[voxel_index[0]-1][voxel_index[1]-1][voxel_index[2]-1]
                else:
                    return np.exp(mu_mass[mat_array[voxel_index[0]-1][voxel_index[1]-1][voxel_index[2]-1]](np.log(energy)))/densities[mat_array[voxel_index[0]-1][voxel_index[1]-1][voxel_index[2]-1]]
        else:
            raise ValueError('parameter \"kind\" must be either \'l\' or \'m\'')
    
    voxel_info = []
    
    beam_energy = np.array(beam_energy)
    ini_fluence = np.array(ini_fluence)
    
    # THIS IS AN INCOMPLETE PART OF THIS FUNCTION: THIS WILL NOT WORK FOR ANY FAN BEAMS EXCEPT THIS ONE CASE
    # this case is a fan beam moving from positive to negative z with spead in y direction
    if angle_spread[1] != 0:
        ini_fluence = ini_fluence*np.pi*position_spread[0]*position_spread[1]
        ini_fluence = ini_fluence/(2*angle_spread[1]*position_spread[0])
    
    if air_index == None:
        for n in range(len(beam_coor)): 
            if (beam_coor[n][0][1]-beam_coor[n][0][0])==0 and (beam_coor[n][1][1]-beam_coor[n][1][0])==0 and (beam_coor[n][2][1]-beam_coor[n][2][0])==0:
                raise ValueError('X-Ray beam cannot have length of 0. Adjust beam_coor parameter.')
            voxel_info.append(TERMA(num_planes,voxel_lengths,beam_coor[n],ini_planes,angle_spread,position_spread,beam_energy,ini_fluence/len(beam_coor),mu,percent_cutoff=percent_cutoff))
    else:
        for n in range(len(beam_coor)): 
            if (beam_coor[n][0][1]-beam_coor[n][0][0])==0 and (beam_coor[n][1][1]-beam_coor[n][1][0])==0 and (beam_coor[n][2][1]-beam_coor[n][2][0])==0:
                raise ValueError('X-Ray beam cannot have length of 0. Adjust beam_coor parameter.')
            voxel_info.append(TERMA(num_planes,voxel_lengths,beam_coor[n],ini_planes,angle_spread,position_spread,beam_energy,ini_fluence/len(beam_coor),mu,percent_cutoff=percent_cutoff,mat_array=mat_array,air_index=air_index))
    
    kernel_arrays = []
    for kernelname in kernelnames:
        if exists(kernelname + '.npy'):
            kernel_array = np.load(kernelname + '.npy')
        else:
            kernel_array_raw = BinnedResult(kernelname).data['Sum'] 
            kernel_array = kernel_array_raw/np.sum(kernel_array_raw) 
            np.save(kernelname,kernel_array)
        kernel_arrays.append(kernel_array)
    
    for index in range(len(kernel_arrays)):
        if index!=0 and np.shape(kernel_arrays[index])!=np.shape(kernel_arrays[index-1]):
            raise ValueError('All kernels must have same dimensions and number of bins.')
    
    # print('Calling Superposition')
    
    energy_deposit = Superposition(kernel_arrays,kernel_size,num_planes,voxel_lengths,voxel_info,beam_coor,eff_distance,mat_array,num_cores)
    return energy_deposit

