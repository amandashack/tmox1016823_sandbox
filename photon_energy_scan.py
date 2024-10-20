import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import arpys
from load_xtc import *
from pyimagetool import imagetool

def plot_hv_scan(ds, tofs_list):
    for i, (arg, ax) in enumerate(zip(args, axes.flatten())):
        dims = list(arg.dims)
        alignment = arg.attrs.get('alignment', 'N/A')
        polarization = arg.attrs.get('polarization', 'N/A')
        sample_name = arg.attrs.get('material_value', 'N/A')
        photon_energy = arg.attrs.get('photon_energy', 'N/A')

        # Downsample if needed
        if "downsample" in kwargs.keys():
            arg_ds = arg.arpes.downsample({dims[2]: kwargs["downsample"]})
            arg_cut = arg_ds.sel({dims[2]: energy}, method='nearest')
        else:
            arg_cut = arg.sel({dims[2]: energy}, method='nearest')

        # Plot with or without colorbar
        if "add_colorbar" in kwargs.keys():
            arg_cut.plot(x=dims[0], y=dims[1], ax=ax, cmap=color, add_colorbar=kwargs['add_colorbar'])
        else:
            arg_cut.plot(x=dims[0], y=dims[1], ax=ax, cmap=color, add_colorbar=False)

        # Add the information text
        info_text = (f"Sample: {sample_name}\nhv: {photon_energy} eV\nAlignment: {alignment}\n"
                      f"Polarization: {polarization}\nBinding Energy: {energy}")
        ax.text(0.95, 0.05, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                horizontalalignment='right')

        # Turn off the title for each axis
        ax.set_title('')
        ni += 1
    for j in range(ni, num_rows * 2):
        fig.delaxes(axes.flatten()[j])
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.98])
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
    imagetool(hv_scan_xar)