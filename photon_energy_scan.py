import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import arpys
from load_xtc import *
from plot_imagetool import Plot_imagetool

def plot_hv_scan(ds, tofs_list):
    pass

if __name__ == "__main__":
    fetch_shots = 1000  # Number of shots to load
    update = 100  # Update progress after how many shots?

    expt = 'tmox1016823'
    run = 6

    run6 = xtc_set(run=run, experiment=expt, max_shots=None)
    run6.load_xtc(electron_roi=(4300, 20000), fix_waveform_baseline=False, plot=False)
    data = run6._waveform
    tof_axis = run6.time_px
    scan_axis = run6.scan_var
    hv_scan_xar = xr.DataArray(data, coords={'tof': tof_axis, 'scan_var': scan_axis}, dims=['tof', 'scan_var'])
    Plot_imagetool(hv_scan_xar)