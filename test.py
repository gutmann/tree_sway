#!/usr/bin/env python

import os
import sys
import time

import numpy as np
from scipy.signal import tukey
from scipy.ndimage.filters import median_filter as medfilt
from matplotlib import pyplot as plt

from interception import video_reader as vr

# import mygis
# import xarray as xr


if len(sys.argv)<2:
    print("Usage: test.py <filename.mp4>")
    sys.exit()
else:
    filename = sys.argv[1]

normalize = True

min_amplitude = 0.001
min_freq = 0.2
max_freq = 2.5

reduction = 2
max_times = 50
start_time = 2

plt.figure(figsize=(20,10))


file_base = filename.split(".")[0]
prefix = "t{}_{}_".format(max_times, start_time)


if os.path.isfile("{prefix}amplitudes_{}.png".format(file_base, prefix=prefix)):
    print("Already processed: "+filename)
    sys.exit()

video = vr.Video_Reader(filename)

nx = int(video.shape[2]/reduction)
ny = int(video.shape[1]/reduction)

print("loading data")
t0 = time.time()

frame_rate = video.metadata["r_frame_rate"]
fps = float(frame_rate.split("/")[0]) / float(frame_rate.split("/")[1])
print("FPS:{}".format(fps))
max_times *= fps
start_time *= fps
max_times = int(max_times)
start_time = int(start_time)

data = np.zeros((min(video.shape[0],max_times), ny, nx), dtype=np.float)

if start_time > 0:
    for i in range(start_time):
        _ = video.next()

for i,v in enumerate(video):
    if i==0:
        initial_image = v[:,:,:]

    if i<data.shape[0]:
        if reduction > 1:
            data[i,:,:] = v[:ny*reduction, :nx*reduction, 2].reshape((ny,reduction, nx, reduction)).mean(axis=1).mean(axis=2)
        else:
            data[i,:,:] = v[:,:,2]
    else:
        break
print("finished: {:5.3} seconds\n".format(time.time()-t0))

print("plotting initial image")
t0 = time.time()
plt.imshow(initial_image, origin="upper")
plt.savefig("{prefix}image_{}.png".format(file_base, prefix=prefix))
print("finished: {:5.3} seconds\n".format(time.time()-t0))

print("computing fft")
t0 = time.time()
nt = min(max_times, min(data.shape[0], i))
print(data.shape[0], nt, max_times)
window = tukey(nt,0.1)
data = data[:nt,:,:]# * window[:,np.newaxis,np.newaxis]
f_data = np.zeros(data.shape, dtype=np.float)

if normalize:
    data_mean = data.mean(axis=0)
    for i in range(nt):
        data[i] -= data_mean

# data = np.transpose(data, (1,2,0))

# for memory efficiency purposes
for i in range(data.shape[1]):
    f_data[:,i,:] = np.abs(np.fft.fft(data[:,i,:], axis=0)) / nt

print("finished: {:5.3} seconds\n".format(time.time()-t0))

print("finding best frequencies")
t0 = time.time()
freqs = np.fft.fftfreq(nt, 1/fps)
print(freqs.min(), freqs.max())
bottom = np.where(freqs > min_freq)[0][0]
top = np.where(freqs > max_freq)[0][0]
print(bottom, top)

# xr.DataArray(f_data[bottom:top]).to_netcdf(file_base+".nc")
# mygis.write(file_base+".nc", f_data[bottom:top])

best = np.argmax(f_data[bottom:top], axis=0)
ampl = np.max(f_data[bottom:top], axis=0) / data.mean(axis=0)

def filter_best(data, ampl):
    ny, nx = data.shape
    output = np.zeros(data.shape, dtype=np.int)
    for i in range(1,ny-2):
        for j in range(1,nx-2):
            pos = np.argmax(ampl[i-1:i+2, j-1:j+2])
            output[i,j] = data[i-1:i+2, j-1:j+2].flat[pos]
    return medfilt(output, (3,3)).astype('i')

best2 = filter_best(best, ampl)

freq_data = freqs[best2+bottom]

masked_freq_data = np.ma.array(freq_data, mask = ampl < min_amplitude)
print("finished: {:5.3} seconds\n".format(time.time()-t0))

print("plotting frequency image")
t0 = time.time()
plt.clf();
plt.imshow(masked_freq_data, vmax=2, vmin=0.1, origin="upper", cmap=plt.cm.jet)
plt.colorbar()
plt.savefig("{prefix}fft_{}.png".format(file_base, prefix=prefix))
print("finished: {:5.3} seconds\n".format(time.time()-t0))

print("plotting amplitude image")
t0 = time.time()
plt.clf()
plt.imshow(ampl, origin="upper", vmax=2, vmin=0)
plt.colorbar()
plt.savefig("{prefix}amplitudes_{}.png".format(file_base, prefix=prefix))
print("finished: {:5} seconds\n".format(time.time()-t0))
