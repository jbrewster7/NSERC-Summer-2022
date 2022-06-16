import io
import matplotlib.pyplot as plt
import numpy as np
from mpl_interactions import hyperslicer

def slider(data,vmin=0):
    '''
    Parameters:
    ----------
    data :: array 
      data taken from a topas2numpy BinnedResult
    
    Returns:
    -------
    fig,ax,control :: idk plot stuff ig 
    '''
    fig1, ax1 = plt.subplots()
    control1 = hyperslicer(data, vmin=vmin, vmax=np.max(data), play_buttons=True, play_button_pos="left")
    
    return fig1,ax1,control1