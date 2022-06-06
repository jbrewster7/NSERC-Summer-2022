from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import numpy as np

def update(val):
    image = ax.imshow(a[arr_slider.val],vmin=0,vmax=np.max(a))
    fig.canvas.draw_idle()

def SlidePlot(inix,a):
    fig, ax = plt.subplots()

    image = ax.imshow(a[inix],vmin=0,vmax=np.max(a))

    plt.subplots_adjust(left=0.25, bottom=0.25)

    axarr = plt.axes([0.25, 0.1, 0.65, 0.03])
    arr_slider = Slider(
        ax=axarr,
        label='Array Index',
        valmin=0,
        valmax=len(a)-1,
        valinit=inix,
        valstep=1,
    )

    arr_slider.on_changed(update)

    plt.show()
