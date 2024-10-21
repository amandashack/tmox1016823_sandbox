import numpy as np
import h5py
import psana
import multiprocessing as mp
import os

def process_port(args):
    port_num, expt, run_number, electron_roi, downsample_size, batch_size = args

    # Initialize DataSource
    ds = psana.DataSource(exp=expt, run=run_number)
    run = next(ds.runs())

    # Initialize detectors
    hsd = run.Detector('mrco_hsd')
    sv_detector = run.Detector("hf_w")  # Scan variable detector (e.g., photon energy)

    # Initialize accumulators
    waveform_batch = []
    scan_var_batch = []
    time_px = None

    # Prepare HDF5 file for writing data incrementally
    filename = f"port_{port_num}.h5"
    with h5py.File(filename, 'w') as h5f:
        h5f.attrs['port'] = port_num
        h5f.attrs['experiment'] = expt
        h5f.attrs['run_number'] = run_number

        # We'll create expandable datasets to store the downsampled data
        max_waveform_length = electron_roi[1] - electron_roi[0]
        dset_waveforms = h5f.create_dataset('waveforms', shape=(0, max_waveform_length), maxshape=(None, max_waveform_length))
        dset_scan_vars = h5f.create_dataset('scan_vars', shape=(0,), maxshape=(None,))
        # Time axis remains constant; store it once
        dset_time = h5f.create_dataset('time', shape=(max_waveform_length,), data=np.zeros(max_waveform_length))

        total_downsampled_shots = 0  # Keep track of the total number of downsampled shots

        # Process events
        for evt_num, evt in enumerate(run.events()):
            # Get scan variable
            sv = sv_detector(evt)
            if sv is not None:
                sv_value = sv
            else:
                sv_value = np.nan

            # Get waveform data
            hsd_waveforms = hsd.raw.padded(evt)
            if hsd_waveforms is None:
                continue  # Skip if no data

            # Extract time axis once
            if time_px is None:
                try:
                    time_px_full = 1e6 * hsd_waveforms[port_num]['times'].astype('float')
                    # Ensure electron_roi indices are within bounds
                    if electron_roi[1] > len(time_px_full):
                        electron_roi = (electron_roi[0], len(time_px_full))
                    time_px = time_px_full[electron_roi[0]:electron_roi[1]]
                    # Save time axis to HDF5 dataset
                    dset_time[:] = time_px
                except KeyError:
                    print(f"Port {port_num} not found in data. Skipping.")
                    return

            # Extract waveform
            waveform_full = hsd_waveforms[port_num][0].astype('float')
            if electron_roi[1] > len(waveform_full):
                electron_roi = (electron_roi[0], len(waveform_full))
            tof_waveform = waveform_full[electron_roi[0]:electron_roi[1]]

            # Accumulate data
            waveform_batch.append(tof_waveform)
            scan_var_batch.append(sv_value)

            # Once we have collected 'batch_size' shots, downsample and write to file
            if len(waveform_batch) >= batch_size:
                # Convert batch to arrays
                waveforms = np.array(waveform_batch)
                scan_vars = np.array(scan_var_batch)

                # Downsample waveforms
                num_shots = waveforms.shape[0]
                downsample_factor = max(1, num_shots // downsample_size)
                num_shots_trimmed = (num_shots // downsample_factor) * downsample_factor
                waveforms = waveforms[:num_shots_trimmed]
                scan_vars = scan_vars[:num_shots_trimmed]
                waveforms_reshaped = waveforms.reshape(-1, downsample_factor, waveforms.shape[1])
                downsampled_waveforms = waveforms_reshaped.mean(axis=1)
                downsampled_scan_vars = scan_vars[::downsample_factor]

                # Append downsampled data to HDF5 datasets
                num_downsampled_shots = downsampled_waveforms.shape[0]
                dset_waveforms.resize(total_downsampled_shots + num_downsampled_shots, axis=0)
                dset_waveforms[total_downsampled_shots:total_downsampled_shots + num_downsampled_shots] = downsampled_waveforms
                dset_scan_vars.resize(total_downsampled_shots + num_downsampled_shots, axis=0)
                dset_scan_vars[total_downsampled_shots:total_downsampled_shots + num_downsampled_shots] = downsampled_scan_vars
                total_downsampled_shots += num_downsampled_shots

                # Clear the batch accumulators
                waveform_batch = []
                scan_var_batch = []

        # After processing all events, process any remaining data in the batch
        if waveform_batch:
            # Convert batch to arrays
            waveforms = np.array(waveform_batch)
            scan_vars = np.array(scan_var_batch)

            # Downsample waveforms
            num_shots = waveforms.shape[0]
            downsample_factor = max(1, num_shots // downsample_size)
            num_shots_trimmed = (num_shots // downsample_factor) * downsample_factor
            if num_shots_trimmed == 0:
                # Not enough shots to downsample; use mean of available shots
                downsampled_waveforms = waveforms.mean(axis=0, keepdims=True)
                downsampled_scan_vars = np.array([scan_vars.mean()])
            else:
                waveforms = waveforms[:num_shots_trimmed]
                scan_vars = scan_vars[:num_shots_trimmed]
                waveforms_reshaped = waveforms.reshape(-1, downsample_factor, waveforms.shape[1])
                downsampled_waveforms = waveforms_reshaped.mean(axis=1)
                downsampled_scan_vars = scan_vars[::downsample_factor]

            # Append downsampled data to HDF5 datasets
            num_downsampled_shots = downsampled_waveforms.shape[0]
            dset_waveforms.resize(total_downsampled_shots + num_downsampled_shots, axis=0)
            dset_waveforms[total_downsampled_shots:total_downsampled_shots + num_downsampled_shots] = downsampled_waveforms
            dset_scan_vars.resize(total_downsampled_shots + num_downsampled_shots, axis=0)
            dset_scan_vars[total_downsampled_shots:total_downsampled_shots + num_downsampled_shots] = downsampled_scan_vars
            total_downsampled_shots += num_downsampled_shots

    print(f"Port {port_num}: Processing complete. Total downsampled shots: {total_downsampled_shots}")

if __name__ == "__main__":
    # Parameters
    expt = 'tmox1016823'        # Experiment name
    run_number = 6              # Run number to process
    electron_roi = (4300, 10000)  # Region of interest in the waveform data
    downsample_size = 50        # Number of shots to downsample to
    batch_size = 1000           # Number of shots to process before downsampling

    # List of ports to process
    ports = [0, 1, 2, 3]  # Replace with your actual port numbers

    # Prepare arguments for each process
    args_list = [(port_num, expt, run_number, electron_roi, downsample_size, batch_size) for port_num in ports]

    # Create a pool of workers, one per port
    with mp.Pool(processes=len(ports)) as pool:
        pool.map(process_port, args_list)



