#!/usr/bin/env python

"""
SYNOPSIS

    process_brightness.py [-h] [--verbose] [--plot] [-v, --version] <filename> <outputfile_base_name>

DESCRIPTION

    TODO This describes how to use this script.
    This docstring will be printed by the script if there is an error or
    if the user requests help (-h or --help).

EXAMPLES

    process_brightness.py IMG0092.m4v

AUTHOR

    Ethan Gutmann - gutmann@ucar.edu

LICENSE

    This script is in the public domain.

REQUIREMENTS

    numpy, scipy, interception
    if plotting : matplotlib
    if saving to netcdf : xarray

VERSION
    1.0

"""
from __future__ import absolute_import, print_function, division

import sys
import os
import traceback
import argparse
import time

import numpy as np
from scipy.signal import tukey
from scipy.ndimage.filters import median_filter as medfilt
from matplotlib import pyplot as plt

from interception import video_reader as vr


global verbose
verbose = False

def filter_best(data, ampl):
    ny, nx = data.shape
    output = np.zeros(data.shape, dtype=np.int)
    for i in range(1,ny-2):
        for j in range(1,nx-2):
            pos = np.argmax(ampl[i-1:i+2, j-1:j+2])
            output[i,j] = data[i-1:i+2, j-1:j+2].flat[pos]
    return medfilt(output, (3,3)).astype('i')


def main (filename, output_file,
          plot_data = False,
          save_results = False,
          normalize = True,
          min_amplitude = 0.001,
          min_freq = 0.2,
          max_freq = 2.5,
          reduction = 2,
          max_times = 50,
          start_time = 2):

    file_base = output_file # filename.split(".")[0]
    prefix = "t{}_{}_".format(max_times, start_time)


    if os.path.isfile("{prefix}amplitudes_{}.png".format(file_base, prefix=prefix)):
        if verbose: print("Already processed: "+filename)
        sys.exit()

    video = vr.Video_Reader(filename)

    nx = int(video.shape[2]/reduction)
    ny = int(video.shape[1]/reduction)

    if verbose: print("loading data")
    t0 = time.time()

    frame_rate = video.metadata["r_frame_rate"]
    fps = float(frame_rate.split("/")[0]) / float(frame_rate.split("/")[1])
    if verbose: print("FPS:{}".format(fps))
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

    if plot_data:
        plt.figure(figsize=(20,10))
        if verbose: print("plotting initial image")
        t0 = time.time()
        plt.imshow(initial_image, origin="upper")
        plt.savefig("{prefix}image_{}.png".format(file_base, prefix=prefix))
        if verbose: print("finished: {:5.3} seconds\n".format(time.time()-t0))

    if verbose: print("computing fft")
    t0 = time.time()
    nt = min(max_times, min(data.shape[0], i))
    if verbose: print(data.shape[0], nt, max_times)
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

    if verbose: print("finished: {:5.3} seconds\n".format(time.time()-t0))

    if verbose: print("finding best frequencies")
    t0 = time.time()
    freqs = np.fft.fftfreq(nt, 1/fps)
    if verbose: print(freqs.min(), freqs.max())
    bottom = np.where(freqs > min_freq)[0][0]
    top = np.where(freqs > max_freq)[0][0]
    if verbose: print(bottom, top)

    if save_results:
        import xarray as xr
        xr.DataArray(f_data[bottom:top]).to_netcdf(file_base+".nc")

    best = np.argmax(f_data[bottom:top], axis=0)
    ampl = np.max(f_data[bottom:top], axis=0) / data.mean(axis=0)

    # best = filter_best(best, ampl)

    freq_data = freqs[best+bottom]

    masked_freq_data = np.ma.array(freq_data, mask = ampl < min_amplitude)
    if verbose: print("finished: {:5.3} seconds\n".format(time.time()-t0))

    if plot_data:
        if verbose: print("plotting frequency image")
        t0 = time.time()
        plt.clf();
        plt.imshow(masked_freq_data, vmax=2, vmin=0.1, origin="upper", cmap=plt.cm.jet)
        plt.colorbar()
        plt.savefig("{prefix}fft_{}.png".format(file_base, prefix=prefix))
        if verbose: print("finished: {:5.3} seconds\n".format(time.time()-t0))

        if verbose: print("plotting amplitude image")
        t0 = time.time()
        plt.clf()
        plt.imshow(ampl, origin="upper", vmax=2, vmin=0)
        plt.colorbar()
        plt.savefig("{prefix}amplitudes_{}.png".format(file_base, prefix=prefix))
        if verbose: print("finished: {:5} seconds\n".format(time.time()-t0))


if __name__ == '__main__':
    try:
        parser= argparse.ArgumentParser(description='Compute the dominant brightness frequencies in a video. ',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('filename', action='store', default="movie.mp4", help="Name of movie file to process")
        parser.add_argument('-o',       dest="outputfile",                  action='store',  default="out_", help="prefix of output files")
        parser.add_argument('--mina',   dest="min_amplitude",   nargs="?",  action='store',  default=0.001, type=float, help="minimum amplitude to plot")
        parser.add_argument('--minf',   dest="min_freq",        nargs="?",  action='store',  default=0.2,   type=float, help="minimum frequency to examine")
        parser.add_argument('--maxf',   dest="max_freq",        nargs="?",  action='store',  default=2.5,   type=float, help="maximum frequency to examine")
        parser.add_argument('--reduce', dest="reduction",       nargs="?",  action='store',  default=2,     type=float, help="factor to reduce x/y dimensions by")
        parser.add_argument('--nt',     dest="max_times",       nargs="?",  action='store',  default=50,    type=float, help="maximum time period to examine [s]")
        parser.add_argument('--st',     dest="start_time",      nargs="?",  action='store',  default=2,     type=float, help="start time [s]")
        parser.add_argument('--plot_data',   dest='plot_data',     action='store_true',  default=False, help='verbose output')
        parser.add_argument('--save_results',dest='save_results',  action='store_true',  default=False, help='verbose output')
        parser.add_argument('--normalize',   dest='normalize',     action='store_true',  default=False, help='verbose output')
        parser.add_argument('-v', '--version', action='version', version='Process Brightness 1.0')
        parser.add_argument('--verbose',dest='verbose', action='store_true',  default=False, help='verbose output')
        args = parser.parse_args()

        verbose = args.verbose

        exit_code = main(args.filename, args.outputfile,
                          plot_data = args.plot_data,
                          save_results = args.save_results,
                          normalize = args.normalize,
                          min_amplitude = args.min_amplitude,
                          min_freq = args.min_freq,
                          max_freq = args.max_freq,
                          reduction = args.reduction,
                          max_times = args.max_times,
                          start_time = args.start_time)

        if exit_code is None:
            exit_code = 0
        sys.exit(exit_code)
    except KeyboardInterrupt as e: # Ctrl-C
        raise e
    except SystemExit as e: # sys.exit()
        raise e
    except Exception as e:
        print('ERROR, UNEXPECTED EXCEPTION')
        print(str(e))
        traceback.print_exc()
        os._exit(1)
