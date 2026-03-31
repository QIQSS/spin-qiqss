import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq

from matplotlib.figure import Figure
from matplotlib.axes import Axes

IMKW = dict(aspect="auto", origin="lower", interpolation="none")

def mk_extent(vec1, vec2, center=False):
    if type(vec1) is int:
        vec1 = np.arange(vec1)
    if type(vec2) is int:
        vec2 = np.arange(vec2)
    
    min1, max1 = vec1[0], vec1[-1]
    min2, max2 = vec2[0], vec2[-1]

    if center:
        dx = (min1 - max1) / (len(vec1)-1)
        dy = (min2 - max2) / (len(vec2)-1)
        vec1 = np.linspace(min1 - dx/2, max1 + dx/2, len(vec1))
        vec2 = np.linspace(min2 - dy/2, max2 + dy/2, len(vec2))

    return [*vec1[[0, -1]], *vec2[[0, -1]],]


def hist_2d_with_thresholds(
    hists,
    x_axis,
    thresholds_init = None,
    thresholds_end = None,
) -> tuple[Figure, Axes]:
    """
    Plot the image of histograms, horizontaly.
    Overlay the lines thresholds_*.
    Returns:
        fig:Figure, ax:Axes
    """
    N = hists.shape[0]
    fig, ax = plt.subplots()

    extent = [*x_axis[[0, -1]], 0, N]
    im = ax.imshow(hists, aspect="auto", origin="lower", interpolation="none", extent=extent)
    if thresholds_init is not None:
        ax.plot(thresholds_init, np.arange(N)+0.5, c="y", label="Threshold init")
    if thresholds_end is not None:
        ax.plot(thresholds_end, np.arange(N)+0.5, c="r", label="Threshold readout")
    ax.set_xlabel("Histogram I")
    ax.set_ylabel("Index")
    ax.legend()
    cb = fig.colorbar(im, ax=ax, label="Count")
    return fig, ax

def chevron_time_freq(
    s_to_s,
    time_axis_ns,
    other_axis,
    title="Init Singlet -> Readout Singlet (Ramsey)"
) -> tuple[Figure, Axes, Axes]:
    """
    Plot side by side: chevron with time axis, chevron with freq_axis
    Returns:
        fig: Figure
        ax_im_time: Axes of time plot
        ax_im_freq: Axes of freq plot
    """

    fft_lbl = lambda line: np.abs(rfft(line - np.mean(line)))
    s_to_s_f = np.apply_along_axis(fft_lbl, 1, s_to_s)
    freqs_Mhz = rfftfreq(len(time_axis_ns), np.diff(time_axis_ns)[0]*1e-9)*1e-6


    plt.figure()
    fig, (ax_im_time, ax_im_freq) = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [2, 1]})

    im1 = ax_im_time.imshow(s_to_s, origin="lower", interpolation="none", aspect="auto", extent=mk_extent(time_axis_ns, other_axis))
    fig.colorbar(im1, ax=ax_im_time)

    ax_im_time.set(xlabel="Wait time (ns)")

    im2 = ax_im_freq.imshow(s_to_s_f, origin="lower", interpolation="none", aspect="auto", extent=mk_extent(freqs_Mhz, other_axis))
    ax_im_freq.set(xlabel="Fréquence (Mhz)")
    ax_im_freq.tick_params(labelleft=False)
    #ax_im_freq.set_xlim(0,2)

    fig.suptitle(title)
    fig.tight_layout()

    return fig, ax_im_time, ax_im_freq