from load_xtc import xtc_set
import numpy as np
import xarray as xr



if __name__ == "__main__":
    fetch_shots = 1000  # Number of shots to load
    update = 100  # Update progress after how many shots?
    run_start = 34
    run_stop = 42
    runs = np.arange(run_start, run_stop)

    expt = 'tmox1016823'
    #ports = [202, 247, 292, 270, 135, 180, 315, 90, 22, 225, 67, 45, 112, 157, 0, 337]
    ports = [0]

    run6 = xtc_set(run=runs, experiment=expt, max_shots=fetch_shots, scan=False)

    # Initialize a list to hold the averaged waveforms for each port
    averaged_waveforms_per_port = []
    unique_scan_vars = None  # To store the unique scan variables

    for p in ports:
        # Load data for the current port
        print(f"Processing port {p}")
        run6.load_xtc(electron_roi=(4300, 10000), fix_waveform_baseline=False, port_num=p, plot=False)

        # Extract data from run_dict
        run_data = run6.run_dict[runs]
        tof_axis = run_data['time']
        scan_vars = run_data['scan_var']
        waveforms = run_data['waveform']

        # Create a DataArray for waveforms with coordinates
        wf_da = xr.DataArray(
            waveforms,
            coords={'shot': np.arange(waveforms.shape[0]), 'tof': tof_axis},
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

    # Call the plotting function with data verification enabled
    plot_hv_scan(hv_scan_xar, verify_data=True, num_verify=2)