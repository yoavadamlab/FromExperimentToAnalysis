import numpy as np
from scipy import signal
from scipy import stats    
import cv2

################# spike detection methods #################

######################### Volpy ###########################

def signal_filter(sg, freq, fr, order=3, mode='high'):
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg

def adaptive_thresh(pks, clip, pnorm=0.5, min_spikes=10):
    """ Adaptive threshold method for deciding threshold given heights of all peaks.
    Args:
        pks: 1-d array
            height of all peaks
        clip: int
            maximum number of spikes for producing templates
        pnorm: float, between 0 and 1, default is 0.5
            a variable deciding the amount of spikes chosen for adaptive threshold method
            
        min_spikes: int
            minimal number of spikes to be detected
    Returns:
        thresh: float
            threshold for choosing spikes
        falsePosRate: float
            possibility of misclassify noise as real spikes
        detectionRate: float
            possibility of real spikes being detected
        low_spikes: boolean
            true if number of spikes is smaller than minimal value
    """
    # find median of the kernel density estimation of peak heights
    spread = np.array([pks.min(), pks.max()])
    spread = spread + np.diff(spread) * np.array([-0.05, 0.05])
    low_spikes = False
    pts = np.linspace(spread[0], spread[1], 2001)
    kde = stats.gaussian_kde(pks)
    f = kde(pts)    
    xi = pts
    center = np.where(xi > np.median(pks))[0][0] # the index of the first peak greater than median

    fmodel = np.concatenate([f[0:center + 1], np.flipud(f[0:center])])
    if len(fmodel) < len(f):
        fmodel = np.append(fmodel, np.ones(len(f) - len(fmodel)) * min(fmodel))
    else:
        fmodel = fmodel[0:len(f)]

    # adjust the model so it doesn't exceed the data:
    csf = np.cumsum(f) / np.sum(f)
    csmodel = np.cumsum(fmodel) / np.max([np.sum(f), np.sum(fmodel)])
    lastpt = np.where(np.logical_and(csf[0:-1] > csmodel[0:-1] + np.spacing(1), csf[1:] < csmodel[1:]))[0]
    if not lastpt.size:
        lastpt = center
    else:
        lastpt = lastpt[0]
    fmodel[0:lastpt + 1] = f[0:lastpt + 1]
    fmodel[lastpt:] = np.minimum(fmodel[lastpt:], f[lastpt:])

    # find threshold
    csf = np.cumsum(f)
    csmodel = np.cumsum(fmodel)
    csf2 = csf[-1] - csf
    csmodel2 = csmodel[-1] - csmodel
    obj = csf2 ** pnorm - csmodel2 ** pnorm
    maxind = np.argmax(obj)
    thresh = xi[maxind]

    if np.sum(pks > thresh) < min_spikes:
        low_spikes = True
        print(f'Few spikes were detected. Adjusting threshold to take {min_spikes} largest spikes')
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
    elif ((np.sum(pks > thresh) > clip) & (clip > 0)):
#         print(f'Selecting top {clip} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))

    ix = np.argmin(np.abs(xi - thresh))
    falsePosRate = csmodel2[ix] / csf2[ix]
    detectionRate = (csf2[ix] - csmodel2[ix]) / np.max(csf2 - csmodel2)
    return thresh, falsePosRate, detectionRate, low_spikes

def whitened_matched_filter(data, locs, window):
    """
    Function for using whitened matched filter to the original signal for better
    SNR. Use welch method to approximate the spectral density of the signal.
    Rescale the signal in frequency domain. After scaling, convolve the signal with
    peak-triggered-average to make spikes more prominent.
    
    Args:
        data: 1-d array
            input signal
        locs: 1-d array
            spike times
        window: 1-d array
            window with size of temporal filter
    Returns:
        datafilt: 1-d array
            signal processed after whitened matched filter
    
    """
    N = np.ceil(np.log2(len(data)))
    censor = np.zeros(len(data))
    censor[locs] = 1
    censor = (np.convolve(censor.flatten(), np.ones([1, len(window)]).flatten(), 'same')).astype(int)
    censor = (censor < 0.5)
    noise = data[censor]

    _, pxx = signal.welch(noise, fs=2 * np.pi, window=signal.get_window('hamming', 1000), nfft=2 ** N, detrend=False,
                          nperseg=1000)
    Nf2 = np.concatenate([pxx, np.flipud(pxx[1:-1])])
    scaling_vector = 1 / np.sqrt(Nf2)

    cc = np.pad(data.copy(),(0,int(2**N-len(data))),'constant')    
    dd = (cv2.dft(cc,flags=cv2.DFT_SCALE+cv2.DFT_COMPLEX_OUTPUT)[:,0,:]*scaling_vector[:,np.newaxis])[:,np.newaxis,:]
    dataScaled = cv2.idft(dd)[:,0,0]
    PTDscaled = dataScaled[(locs[:, np.newaxis] + window)]
    PTAscaled = np.mean(PTDscaled, 0)
    datafilt = np.convolve(dataScaled, np.flipud(PTAscaled), 'same')
    datafilt = datafilt[:len(data)]
    return datafilt

def volpy_spike_detection(data, fr=500, pnorm=0.5):
    hp_freq = 1
    clip = 200
    min_spikes = 10
    threshold_method = "adaptive_threshold"
    template_size = 0.02 # ms, size of the spike window
    window_length = int(fr * template_size) # half window length for spike templates
    # high-pass filter the signal for spike detection
    data = signal_filter(data, hp_freq, fr, order=5)
    data = data - np.median(data)
    pks = data[signal.find_peaks(data, height=None)[0]]

    # first round of spike detection    
    if threshold_method == 'adaptive_threshold':
        thresh, _, _, low_spikes = adaptive_thresh(pks, clip, 0.25, min_spikes)
        locs = signal.find_peaks(data, height=thresh)[0]
    elif threshold_method == 'simple':
        thresh, low_spikes = simple_thresh(data, pks, clip, 3.5, min_spikes)
        locs = signal.find_peaks(data, height=thresh)[0]
    else:
        print("Error: threshold_method not found")
        raise Exception('Threshold_method not found!')

    # spike template
    window = (np.arange(-window_length, window_length + 1, 1)).astype(int)
    locs = locs[np.logical_and(locs > (-window[0]), locs < (len(data) - window[-1]))]
    PTD = data[(locs[:, np.newaxis] + window)]
    PTA = np.median(PTD, 0)
    PTA = PTA - np.min(PTA)
    templates = PTA

    # whitened matched filtering based on spike times detected in the first round of spike detection
    datafilt = whitened_matched_filter(data, locs, window)    
    datafilt = datafilt - np.median(datafilt)

    # second round of spike detection on the whitened matched filtered trace
    pks2 = datafilt[signal.find_peaks(datafilt, height=None)[0]]
    if threshold_method == 'adaptive_threshold':
        thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip=0, pnorm=pnorm, min_spikes=min_spikes)  # clip=0 means no clipping
        spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    elif threshold_method == 'simple':
        thresh2, low_spikes = simple_thresh(datafilt, pks2, 0, threshold, min_spikes)
        spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    return spikes, datafilt, thresh2

##################### end of volpy #############################

################### median filter ##############################
def median_filter(signal, window_size):
    filtered_signal = []
    for i in range(len(signal)):
        # Select a window of samples centered at the current sample
        window_start = i - window_size // 2
        window_end = i + window_size // 2 + 1
        if (window_start >= 0) and (window_end <= len(signal)):
            window = signal[window_start:window_end]        
            filtered_signal.append(np.median(window))    
        else:
            if (window_start < 0):
                zeros = np.zeros(np.abs(window_start))
                filtered_signal.append(np.median(np.concatenate([zeros, signal[:window_end]])))
            if (window_end > len(signal)):
                zeros = np.zeros(window_end - len(signal))
                filtered_signal.append(np.median(np.concatenate([signal[window_start:], zeros])))
    return np.array(filtered_signal)

def hp_flter_median_method(trace):
    window_size = 20 # 10 data points from each side for calculating median of a point
    if isinstance(trace, np.ndarray):
        trace = trace
    else:
        trace = trace.to_numpy()
    filtered_signal = median_filter(trace, window_size) 
    median_filterdd = trace - filtered_signal
#     median_filterdd = median_filterdd * [median_filterdd >0]
    # median_filterdd = median_filterdd * [median_filterdd <0.1]
    median_filterdd = median_filterdd.flatten()
    return median_filterdd

def median_filter_detection(trace, fr, std_num):
    hp_trace = hp_flter_median_method(trace)
    th = hp_trace.mean() + std_num * hp_trace.std()
    spikes_time = signal.find_peaks(hp_trace, height=th, distance=2)[0]
    return spikes_time, hp_trace, th

################# end of median filter #########################