import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import arpys
from load_xtc import *
from pyimagetool import imagetool


def plot_hv_scan(hv_scan_xar):
    ports = hv_scan_xar.coords['ports'].values
    num_ports = len(ports)

    # Determine the number of columns and rows for subplots
    num_cols = 4  # Adjust as needed
    num_rows = (num_ports + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))
    axes = axes.flatten()

    for i, port in enumerate(ports):
        data = hv_scan_xar.sel(ports=port)
        ax = axes[i]

        # Transpose data to match the axes
        pcm = ax.pcolormesh(data.coords['scan_var'], data.coords['tof'], data.T, shading='auto')
        ax.set_title(f'Port {port}')
        ax.set_xlabel('Scan Variable')
        ax.set_ylabel('Time of Flight')
        fig.colorbar(pcm, ax=ax)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fetch_shots = 1000  # Number of shots to load
    update = 100  # Update progress after how many shots?

    expt = 'tmox1016823'
    run = 6
    ports = [202, 247, 292, 270, 135, 180, 315, 90, 22, 225, 67, 45, 112, 157, 0, 337]

    run6 = xtc_set(run=run, experiment=expt, max_shots=None)
    wf_ports = []
    for p in ports:
        run6.load_xtc(electron_roi=(4300, 20000), fix_waveform_baseline=False, port_num=p, plot=False)
        wf_ports.append(run6._waveform)
    tof_axis = run6.time_px
    scan_axis = run6.scan_var
    hv_scan_xar = xr.DataArray(wf_ports, coords={'tof': tof_axis, 'scan_var': scan_axis, 'ports': ports},
                               dims=['tof', 'scan_var', 'ports'])
    plot_hv_scan(hv_scan_xar)