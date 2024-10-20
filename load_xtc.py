import psana as ps
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


fwhm_factor = 2 * np.sqrt(2 * np.log(2))

def gaussian(x, A, x0, sigma, c): return A * np.exp(-(0.5 * ((x - x0) ** 2)) / (sigma ** 2)) + c


def nearest_neighbour_interp(arr): return np.array([0.5 * (arr[i] + arr[i + 1]) for i in range(len(arr) - 1)])


def linear_func(x, a, b): return a * x + b


def grad_func(x, a): return a * x


def invert_quad(x, a, b): return a * (x ** (-2)) + b


def bls_fzp(fzps, d=400, verbose=False):
    '''FZP Pirahna baseline subtraction function
    Takes:
    fzps - list of fzps to be processed
    d=400 - distance from edge of detector to use to calculate baseline
    verbose=False - verbose flag
    Returns:
    fzps_bls - fzps baseline subtracted
    baseline - baseline'''

    x = np.concatenate((np.arange(0, d), np.arange(fzps.shape[1] - d, fzps.shape[1])))
    A = np.concatenate((x[:, None], np.ones((len(x), 1))), axis=1)
    A_inv = np.linalg.pinv(A)
    bg = fzps.mean(0)[x]
    fit = A_inv @ bg

    A_full = np.concatenate((np.arange(fzps.shape[1])[:, None], np.ones((fzps.shape[1], 1))), axis=1)
    baseline = A_full @ fit

    fzps_bls = fzps - baseline

    if verbose:
        plt.figure()
        plt.plot(fzps.mean(0), label='Mean FZP Spectrum')
        plt.plot(baseline, linestyle='--', label='Baseline')
        plt.xlabel('Pixel')
        plt.legend()
        plt.grid()
        plt.show()

    return fzps_bls, baseline


def bin_fzp(fzps, pe_bin_size):
    '''FZP Pirahna binning function
    Takes:
    fzps
    pe_bin_size
    Returns:
    fzps_binned'''

    fzps_binned = np.array([fzps[:, i:(i + pe_bin_size)].mean(1) for i in np.arange(0, fzps.shape[1], pe_bin_size)]).T
    return fzps_binned


def time_to_mq(t, t_0=8335.33533534, alpha=3.7975566284858645e-07):
    '''Calibration for ion-TOF spectrometer found using data from 5th november 2023. returns mass to charge ratio.
    t_0: light peak, alpha: found from argon ionisation series
    returns mass to charge ratio
    '''
    return alpha * (t - t_0) ** 2


def hit_finding(tof_matrix, time_samples, times, time_roi_min=0, time_roi_max=4, height=15, prominence=5):
    '''Conducts hit finding using scipy function. tof matrix must have each shot as rows along axis 0.
    Time samples gives time resolution of output spectrum, an integer.
    Times is the array of times corresponding to each pixel.
    Time ROI should be given and include all visible events. Values height and prominence may need changing'''
    number_of_shots, _ = tof_matrix.shape
    hit_event_times = times[np.concatenate(
        [find_peaks(tof_matrix[idx], height=height, prominence=prominence)[0] for idx in range(number_of_shots)])]
    hit_time_hist, bin_edges = np.histogram(hit_event_times, bins=np.linspace(time_roi_min, time_roi_max, time_samples))
    time_bin_centres = nearest_neighbour_interp(bin_edges)
    hits_per_shot = len(hit_event_times) / number_of_shots
    return hit_time_hist, time_bin_centres, hits_per_shot


def px2eV(i_px, fzp='nitrogen'):
    '''Uses calibration from Jun on 11/1/23 for nitrogen to calulate photon energy axis
    Input:
    i_px - pixel axis
    fzp='nitrogen' - zone plate inserted
    returns: pe_bins'''
    if fzp == 'nitrogen':
        slope = 0.03265  # eV/px
        pe_bins = slope * i_px + 375.33
    else:
        raise Exception('No calibration for given zone plate')
    return pe_bins


def nth_moment(x, y, mean, n):
    x_repeat = np.array([x] * len(y))
    mean_expanded = np.array([mean] * x_repeat.shape[1]).T
    return np.sum(y * (x_repeat - mean_expanded) ** n, axis=1) / np.sum(y, axis=1)


def find_index(arr, val):
    return np.abs(arr - val).argmin()


def find_indices(arr, vals):
    try:
        return np.array([np.abs(arr - val).argmin() for val in vals])
    except:
        return np.abs(arr - vals).argmin()


def bin_bool_ar(start, stop, n_bins, bin_on):
    bin_edges = np.linspace(start, stop, n_bins + 1)
    lowers, uppers = bin_edges[:-1], bin_edges[1:]
    bin_cents = (lowers + uppers) / 2
    bool_ar = np.zeros((len(lowers), len(bin_on)), dtype=bool)
    for idx, (lower, upper) in enumerate(zip(lowers, uppers)):
        bool_ar[idx] = (bin_on >= lower) & (bin_on <= upper)
    return bin_cents, bool_ar


def clean_itof(wf, bg_roi):
    wf_proc = wf.reshape(-1, 4)
    wf_proc = np.mean(wf_proc[bg_roi[0]:bg_roi[1]], axis=0) - wf_proc
    wf_proc = wf_proc.reshape(-1)
    return wf_proc


def fix_wf_baseline(hsd_in, bgfrom=500 * 64):
    hsd_out = np.copy(hsd_in)
    for i in range(4):
        hsd_out[i::4] -= hsd_out[bgfrom + i::4].mean()
    # for i in (12, 13, 12+32, 12+32):
    for i in (12, 13, 12 + 32, 13 + 32, 25, 26, 25 + 32, 26 + 32):
        hsd_out[i::64] -= hsd_out[bgfrom + i::64].mean()
        hsd_out = -hsd_out
    return hsd_out

class xtc_set:
    ### this is from Felix
    def __init__(self, run, experiment, max_shots=None, scan=False):
        self.run = run
        self.experiment = experiment
        self.no_shots = max_shots
        self.scan = scan
        self.hsd_flag = False

    def load_xtc(self, electron_roi=(5000, 20000), fix_waveform_baseline=False, port_num=0, plot=False, downsample=3):
        ds = ps.DataSource(exp=self.experiment, run=self.run)
        self.electron_roi = electron_roi
        self.fix_waveform_baseline = fix_waveform_baseline
        _fzp, _step_arr, _xgmd, _gmd = [], [], [], []
        waveform_arr = []
        scan_var = []
        Nfound = 0
        Nbad = 0

        for run in ds.runs():
            det_names = [name for name in run.detnames]
            if 'mrco_hsd' in det_names: self.hsd_flag = True  # find if there is hsd data to parse
            if 'gmd' in det_names:
                gmd = run.Detector("gmd")
            else:
                gmd = None
            if 'xgmd' in det_names:
                xgmd = run.Detector("xgmd")
            else:
                xgmd = None
            # sp1k4 = run.Detector("sp1k4")
            if "tmo_fzppiranha" in det_names:
                fzp = run.Detector("tmo_fzppiranha")
            else:
                fzp=None
            if self.scan:
                sv_detector = run.Detector("hf_w")

            if self.hsd_flag:
                hsd = run.Detector('mrco_hsd')

            for nevent, event in enumerate(run.events()):
                if self.no_shots and nevent == self.no_shots: break  # kill loop after desired number of shots
                if gmd:
                    gmd_energy = gmd.raw.milliJoulesPerPulse(event)
                    if xgmd_energy is not None:
                        _gmd.append(gmd_energy)
                if xgmd:
                    xgmd_energy = xgmd.raw.milliJoulesPerPulse(event)
                    if xgmd_energy is not None:
                        _xgmd.append(xgmd_energy)
                # sp1k4 = sp1k4.raw.value(event)
                if fzp:
                    fzp_im = fzp.raw.raw(event)
                    if fzp_im is not None:
                        _fzp.append(fzp_im)

                if self.hsd_flag:
                    #hsd_waveforms = hsd.raw.waveforms(event)
                    hsd_waveforms = (hsd.raw.padded(event))
                    if hsd_waveforms is None:
                        Nbad += 1
                    else:
                        if nevent == 0:
                            self.time_px = 1e6 * hsd_waveforms[port_num]['times'].astype('float')
                            #electron_roi_index = find_indices(self.time_px, self.electron_roi)

                        if self.fix_waveform_baseline:
                            tof_waveform = fix_wf_baseline(hsd_waveforms[port_num][0].astype('float'))[
                                                  self.electron_roi[0]:self.electron_roi[1]]

                        else:
                            tof_waveform = hsd_waveforms[port_num][0].astype('float')[self.electron_roi[0]:self.electron_roi[1]]
                            if plot:
                                print(f"Plotting! Remember, there will be {self.no_shots} plots created based on input to max_shots")
                                fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=300)
                                ax.plot(hsd_waveforms[port_num][0], label='Full Electron Spectra', lw=2, c='maroon')
                                ax.set(xlabel='TOF (μs)', yticks=[])
                                axt = ax.twinx()
                                axt.plot(tof_waveform, label='Cropped Electron spectra', c='dodgerblue', lw=2)
                                axt.set(yticks=[])

                                lines1, labels1 = ax.get_legend_handles_labels()
                                lines2, labels2 = axt.get_legend_handles_labels()

                                axt.legend(lines1 + lines2, labels1 + labels2, facecolor='white', framealpha=1, fontsize=10)
                                plt.show()

                    waveform_arr.append(tof_waveform)
                if self.scan:
                    sv = sv_detector(event)
                    scan_var.append(sv)

                Nfound += 1
            break

        self.fzp = np.array(_fzp)
        self.gmd = np.array(_gmd)
        self.xgmd = np.array(_xgmd)
        self.scan_var = np.array(scan_var)
        self.no_shots = Nfound

        if self.hsd_flag:
            # times in us
            print("Make the waveform please!")
            self.time_px = self.time_px[self.electron_roi[0]:self.electron_roi[1]]
            self._waveform = np.array(waveform_arr)

    """def scan_run(self):
        self.scan = True
        ds = ps.DataSource(exp=self.experiment, run=self.run)
        _step_arr, _step_z_arr = [], []
        for run in ds.runs():
            step_z = run.Detector('tmo_lamp_magnet_z')
            step = run.Detector('step_value')

            for nevent, event in enumerate(run.events()):
                if self.no_shots and nevent == self.no_shots: break  # kill loop after desired number of shots

                _step = step(event)
                _step_arr.append(_step)

                _step_z = step_z(event)
                _step_z_arr.append(_step_z)

        self.scan_var = np.array(_step_arr)
        self.scan_var_z = np.array(_step_z_arr)"""

    def purge_bad_data(self):
        if self.scan_var is not None:
            mask = np.ones_like(self.scan_var, dtype=bool)
        else:
            raise ValueError("self.scan_var is None, cannot create a mask.")

        # Function to update the mask safely
        def update_mask(mask, var):
            if var is not None:
                # Ensure the variable has the same shape as the mask
                if var.shape == mask.shape:
                    mask &= (var != None)  # You might want to use a different condition
                else:
                    raise ValueError(f"Shape mismatch: var has shape {var.shape}, expected {mask.shape}.")
            return mask

        # Update the mask with each variable
        mask = update_mask(mask, self.gmd)
        mask = update_mask(mask, self.xgmd)
        mask = update_mask(mask, self.scan_var)
        self.gmd = self.gmd[mask]
        self.xgmd = self.xgmd[mask]
        self.fzp = self.fzp[mask]
        self.scan_var = self.scan_var[mask]
        no_bad_shots = len(mask) - mask.sum()
        self.no_shots = mask.sum()
        print('{} bad shots purged, {} shots processed'.format(no_bad_shots, self.no_shots))

    def roi(self, coms_roi=None, sums_roi=None):
        self.coms_roi = coms_roi
        self.sums_roi = sums_roi

    def spectrum_process(self, roi_type='px', roi=None, sums_roi_type='px', process_roi=None):
        self.px = np.arange(len(self.fzp[0]))
        _hv_ = self.px
        self.roi = roi
        self.process_roi = process_roi
        self.fzp, _ = bls_fzp(self.fzp, d=self.roi[0])
        self.hv = None
        if self.roi:

            if roi_type == 'px':
                self.fzp = self.fzp[:, self.roi[0]:self.roi[1]]
                self.px = self.px[self.roi[0]: self.roi[1]]

            elif roi_type == 'ev':
                self.hv = px2eV(self.px)
                _idx_roi_ = find_index(self.hv, self.roi[0]), find_index(self.hv, self.roi[1])
                self.fzp = self.fzp[:, _idx_roi_[0]:_idx_roi_[1]]
                self.hv = self.hv[_idx_roi_[0]:_idx_roi_[1]]
                self.px = self.px[_idx_roi_[0]:_idx_roi_[1]]

            else:
                raise Exception('no roi selected')

        if self.hv is not None: _hv_ = self.hv

        if self.process_roi is not None:
            if sums_roi_type == 'px':

                self.index_roi = find_index(self.px, self.process_roi[0]), find_index(self.px, self.process_roi[1])
                self.sums = self.fzp[:, self.index_roi[0]: self.index_roi[1]].sum(axis=1)
                self.coms = nth_moment(self.px[self.index_roi[0]: self.index_roi[1]],
                                       self.fzp[:, self.index_roi[0]: self.index_roi[1]], 0, 1)
                self.fwhm = fwhm_factor * nth_moment(self.px[self.index_roi[0]: self.index_roi[1]],
                                                     self.fzp[:, self.index_roi[0]: self.index_roi[1]], self.coms, 2)

            elif sums_roi_type == 'ev':

                self.index_roi = find_index(_hv_, self.process_roi[0]), find_index(_hv_, self.process_roi[1])
                self.sums = self.fzp[:, self.index_roi[0]: self.index_roi[1]].sum(axis=1)
                self.coms = nth_moment(_hv_[self.index_roi[0]: self.index_roi[1]],
                                       self.fzp[:, self.index_roi[0]: self.index_roi[1]], 0, 1)
                self.fwhm = fwhm_factor * nth_moment(_hv_[self.index_roi[0]: self.index_roi[1]],
                                                     self.fzp[:, self.index_roi[0]: self.index_roi[1]], self.coms, 2)

        else:
            self.sums = self.fzp.sum(axis=1)

        self.gmd = 1e3 * self.gmd
        self.xgmd = 1e3 * self.xgmd

    def tofs_process(self, peak_height=10, peak_prominence=30, time_samples=1000, clean_data=True,
                     clean_electron_roi=(0, 1)):
        self.clean_electron_roi = np.array(clean_electron_roi)
        self.clean_data = clean_data
        self.peak_height = peak_height
        self.peak_prominence = peak_prominence

        clean_electron_index = find_indices(self.time_px, self.clean_electron_roi)
        self._waveform = self._waveform[
                                   clean_electron_index[0]:clean_electron_index[1]].mean() - self._waveform

        # hit finding, electron tof
        self.hit_times, self.time_bin_centres, self.hits_per_shot = hit_finding(self._waveform,
                                                                                time_samples=time_samples,
                                                                                times=self.time_px,
                                                                                time_roi_min=self.time_px.min(),
                                                                                time_roi_max=self.time_px.max(),
                                                                                height=self.peak_height,
                                                                                prominence=self.peak_prominence)

if __name__ == "__main__":
    fetch_shots = 1000  # Number of shots to load
    update = 100  # Update progress after how many shots?

    expt = 'tmox1016823'
    run = 6

    run5 = xtc_set(run=run, experiment=expt, max_shots=5)
    run5.load_xtc(electron_roi=(4300, 20000), fix_waveform_baseline=False, plot=True)
    print("purge bad data -- might not need this anymore")
    run5.purge_bad_data()
    #if len(run5.time_px) % 4 == 0:
    #    run5._waveform = run5._waveform.reshape(run5.no_shots, -1, 4).mean(2)
    #    run5.time_px = run5.time_px.reshape(-1, 4).mean(1)
    #else:
    #    run5._waveform = run5._waveform[:, :-(len(run5.time_px) % 4)].reshape(
    #        run5.no_shots, -1, 4).mean(2)
    #    run5.time_px = run5.time_px[:-(len(run5.time_px) % 4)].reshape(-1, 4).mean(1)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=300)
    ax.plot(run5.time_px, run5._waveform.mean(0), label='Ion spectra', lw=2, c='maroon')
    ax.set(xlabel='TOF (μs)', yticks=[])
    plt.legend("mean TOF")
    plt.show()
    #axt = ax.twinx()
    #axt.plot(run5.mbes_time_px, run5._waveform.mean(0), label='Electron spectra', c='dodgerblue', lw=2)
    #axt.set(yticks=[])

    #lines1, labels1 = ax.get_legend_handles_labels()
    #lines2, labels2 = axt.get_legend_handles_labels()

    #axt.legend(lines1 + lines2, labels1 + labels2, facecolor='white', framealpha=1, fontsize=10)
