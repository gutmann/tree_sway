#!/usr/bin/env python
# Note : on cheyenne/casper "module load ffmpeg" before running
# Note this is a simple FFT analysis.  Need to perform this on the longer time sequence from the original video (maybe averaged to 1080p first)
#  run the fft on 20s windows throughout the time series to compute variations over timeself.
#  also run ffts with 1-24 frames removed from the one end of the segment to find the optimal window.
#  alternatively, try the lomb-scargle routine... doesn't seem to work on arrays at the moment, so maybe pick grid cells


from scipy import signal

import numpy as np
import matplotlib.pyplot as plt

# import mygis

red=0; green=1; blue=2

n=1440
fps=30
filename = "./IMG_8773.MOV"; nx=1920; ny=1080; nc=3

# filename = "./wp2.mov"; nx=272; ny=366; nc=3
# filename = "./WP-big.mov"; nx=1880; ny=562; nc=3

n = int(np.round(85*fps))


try:
    print("Trying to loading netcdf data")
    f = mygis.read_nc("tree_sway_frequency_test.nc").data
except:

    print("Loading video")
    from interception import video_reader
    vid = video_reader.Video_Reader(filename, resolution=(ny,nx,nc))

    d = np.zeros((n, ny, nx))
    for i,v in enumerate(vid):
        print(i, n)

        if (i<n):
            d[i,:,:] = (v[:,:,green])
        else:
            print("More Frames!")
            break

    d = d.transpose([1,2,0])

    # average two frames
    # fps = fps/2
    # d = d[:,:,1::2] + d[:,:,:-1:2]


    print("computing fft")
    f = np.fft.rfft(d,axis=2)

    print("writing netcdf")
    # mygis.write("tree_sway_frequency_test.nc",f)

freq = np.fft.rfftfreq(n, 1/fps)
bottom = np.where(freq>0.1)[0][0]
top = np.where(freq>3)[0][0]

print("finding strongest frequency")
fmx = np.argmax(np.abs(f[:,:,bottom:top]),axis=2)
ampl = np.max(np.abs(f[:,:,bottom:top]),axis=2)

bf = freq[bottom:top][fmx]

# bf = signal.medfilt2d(bf)

# clf();imshow(bf,origin="upper",cmap=cm.jet);colorbar();clim(0.5,1.2)
# clf();imshow(ampl,origin="upper",cmap=cm.jet);colorbar();clim(200,2000)
bfm = np.ma.array(bf, mask= ~((ampl>300)&(bf<2)&(bf>0.2)) )

print("plotting")
plt.figure(figsize=(13,6))
plt.imshow(bfm, origin="upper",cmap=plt.cm.jet)

plt.clim(0.4,1.0)
cbar = plt.colorbar()
cbar.set_label("Frequency [Hz]")
plt.title("Tree sway frequency")
plt.xlabel("x-pixel position")
plt.ylabel("y-pixel position")
plt.tight_layout()
plt.savefig("tree_sway_frequency.png",dpi=200)
