import matplotlib.pyplot as plt
import numpy as np
import bokeh
from bokeh.plotting import output_file, figure, show
from bokeh.models import LinearAxis, Range1d



def print_Utility(isBackward, hs, base, minX, maxX, w):
    xs = np.arange(minX, maxX, w)
    for h in hs:
        if isBackward:
            ys = -np.exp(-base * xs - h)
        else:
            ys = -np.exp(-base * xs + h)
        plt.plot(xs, ys)
    plt.show()
    return


def print_Runtime(time_back, time_forward):

    assert len(time_back) == len(time_forward)
    x = [i + 1 for i in range(len(time_back))]
    y = time_back
    y2 = time_forward
    p = figure(x_range=(0, len(x) + 1))
    p.line(x, y, line_width=2)
    p.circle(x, y, fill_color="white", size=8)

    p.line(x, y2, line_width=2)
    p.circle(x, y2, fill_color="white", size=8)

    show(p)
    return

def animate_StockandAlpha(close_price, forward, backward):
    n = len(close_price)


