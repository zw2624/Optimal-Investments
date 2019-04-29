import matplotlib.pyplot as plt
import numpy as np
import bokeh
from bokeh.plotting import output_file, figure, show, save
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


def print_Runtime(t_list, pos = 'top_left'):
    y = t_list[0][0]
    y2 = t_list[1][0]
    t1 = t_list[0][1]
    t2 = t_list[1][1]
    x = [i + 1 for i in range(len(y))]
    p = figure(x_range=(0, len(x) + 1))
    p.xaxis.axis_label = "Period"
    p.yaxis.axis_label = "Running Time"
    p.line(x, y, line_width=2, legend=t1, line_color="orange")
    p.circle(x, y, legend=t1, fill_color="white", size=8)
    p.line(x, y2, line_width=2, legend=t2, line_color="blue")
    p.circle(x, y2, legend=t2,fill_color="white", size=8)
    if len(t_list) == 3:
        y3 = t_list[2][0]
        t3 = t_list[2][1]
        p.line(x, y3, line_width=2, legend=t3, line_color="green")
        p.circle(x, y3, legend=t3, fill_color="white", size=8)
    p.legend.location = pos
    show(p)
    return

def print_Earning(e_list):
    earning_1 = e_list[0][0]
    earning_2 = e_list[1][0]
    t1 = e_list[0][1]
    t2 = e_list[1][1]
    assert len(earning_1) == len(earning_2)
    x = [i + 1 for i in range(len(earning_1))]
    return_1 = np.cumsum(earning_1)
    return_2 = np.cumsum(earning_2)
    p = figure(x_range=(0, len(x) + 1), title = "Cumulative Earnings")
    p.line(x, return_2, line_width=2, legend=t2, line_color="orange")
    p.line(x, return_1, line_width=2, legend=t1, line_color="blue")
    if len(e_list) == 3:
        return_3 = np.cumsum(e_list[2][0])
        t3 = e_list[2][1]
        p.line(x, return_3, line_width=2, legend=t3, line_color="green")
    p.legend.location = "top_left"
    show(p)
    return




def animate_StockandAlpha(close_price, forward, backward):
    n = len(close_price)


