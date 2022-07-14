def change_view(form,arr):
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
        
    return newarr
