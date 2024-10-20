import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
#import arpys
from load_xtc import *
#from pyimagetool import imagetool


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
    fetch_shots = 100  # Number of shots to load
    update = 100  # Update progress after how many shots?

    expt = 'tmox1016823'
    run = 6
    #ports = [202, 247, 292, 270, 135, 180, 315, 90, 22, 225, 67, 45, 112, 157, 0, 337]
    ports = [0, 90, 180, 270]

    run6 = xtc_set(run=run, experiment=expt, max_shots=fetch_shots, scan=True)

    # Initialize a list to hold the averaged waveforms for each port
    averaged_waveforms_per_port = []
    unique_scan_vars = None  # To store the unique scan variables

    for p in ports:
        # Load data for the current port
        print(f"Processing port {p}")
        run6.load_xtc(electron_roi=(4300, 10000), fix_waveform_baseline=False, port_num=p, plot=False)

        # Get the waveforms and scan variables
        tof_axis = run6.time_px  # Time of flight axis
        waveforms = run6._waveform  # Shape: (num_shots, num_tof_points)
        scan_vars = run6.scan_var   # Shape: (num_shots,)
        print(scan_vars)

        # Create a DataArray for waveforms with coordinates
        wf_da = xr.DataArray(
            waveforms,
            coords={'shot': np.arange(waveforms.shape[1]), 'tof': tof_axis},
            dims=['shot', 'tof']
        )
        # Add scan variable as a coordinate
        wf_da = wf_da.assign_coords(scan_var=('shot', scan_vars))

        # Group by scan variable and compute the mean waveform for each unique scan variable
        averaged_wf = wf_da.groupby('scan_var').mean(dim='shot')

        # Store the averaged waveforms
        averaged_waveforms_per_port.append(averaged_wf.values)

        # Store the unique scan variables (assumed same for all ports)
        if unique_scan_vars is None:
            unique_scan_vars = averaged_wf.coords['scan_var'].values

    # Convert the list of averaged waveforms to a NumPy array
    # Shape: (num_ports, num_unique_scan_vars, num_tof_points)
    averaged_waveforms_per_port = np.array(averaged_waveforms_per_port)

    # Create the xarray DataArray
    hv_scan_xar = xr.DataArray(
        averaged_waveforms_per_port,
        coords={
            'ports': ports,
            'scan_var': unique_scan_vars,
            'tof': tof_axis
        },
        dims=['ports', 'scan_var', 'tof']
    )

    # Call the plotting function
    plot_hv_scan(hv_scan_xar)