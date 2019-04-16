import matplotlib.pyplot as plt
import numpy as np
import bokeh
from bokeh.plotting import output_file, figure, show
from bokeh.models import LinearAxis, Range1d
from bokeh.palettes import Dark2_5 as palette
import itertools



def print_Utility(isBackward, hs, base, minX, maxX, w):
    xs = np.arange(minX, maxX, w)
    colors = itertools.cycle(palette)
    label = "Backward Utiliy" if isBackward else "Forward Utility"
    p = figure(x_range=(minX, maxX), title = label)
    for i, color in zip(range(len(hs)), colors):
        h = hs[i]
        if isBackward:
            ys = -np.exp(-base * xs - h)
        else:
            ys = -np.exp(-base * xs + h)
        p.line(xs, ys, legend="Period = {}".format(i+1), line_color=color)
    p.legend.location = "bottom_right"
    show(p)
    return


def print_Runtime(time_back, time_forward):
    assert len(time_back) == len(time_forward)
    x = [i + 1 for i in range(len(time_back))]
    y = time_back[::-1]
    y2 = time_forward
    p = figure(x_range=(0, len(x) + 1))
    p.xaxis.axis_label = "Period"
    p.yaxis.axis_label = "Running Time"
    p.line(x, y, line_width=2, legend="Backward Method", line_color="orange")
    p.circle(x, y, legend="Backward Method",fill_color="white", size=8)
    p.line(x, y2, line_width=2, legend="Forward Method", line_color="blue")
    p.circle(x, y2, legend="Forward Method",fill_color="white", size=8)
    p.legend.location = "top_left"
    show(p)
    return

def print_Earning(earning_f, earning_b):
    assert len(earning_f) == len(earning_b)
    x = [i + 1 for i in range(len(earning_f))]
    return_f = np.cumsum(earning_f)
    return_b = np.cumsum(earning_b)
    p = figure(x_range=(0, len(x) + 1), title = "Cumulative Earnings")
    p.line(x, return_b, line_width=2, legend="Backward Method", line_color="orange")
    p.line(x, return_f, line_width=2, legend="Forward Method", line_color="blue")
    show(p)
    return




def animate_StockandAlpha(close_price, forward, backward):
    n = len(close_price)


