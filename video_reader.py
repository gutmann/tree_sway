import numpy as np
import subprocess as sp
import signal
import json

FFMPEG = "ffmpeg"
BUFFER_SIZE = 10**8

FFPROBE = "ffprobe"
# ffprobe IMG_9499.MOV  2>&1 | grep Stream | grep Video | awk '{ print $11 }'
# ffprobe -v quiet -print_format json -show_format -show_streams filename
# json.loads(encoded_hand)

class Video_Reader(object):
    """docstring for Video_Reader"""
    command = None
    pipe = None
    resolution = None
    filename = ""
    metadata = None

    def __init__(self, filename, resolution=None):
        super(Video_Reader, self).__init__()

        self.filename = filename

        self.get_metadata()

        self.command = [ FFMPEG,
                    '-i', filename,
                    '-f', 'image2pipe',
                    '-pix_fmt', 'rgb24',
                    '-vcodec', 'rawvideo', '-']
        # note: stdout pipes the data we want
        # stderr captures the status updates ffmpeg would print to the screen
        # stdin prevents ffmpeg from capturing keyboard input.
        self.pipe = sp.Popen(self.command, bufsize=BUFFER_SIZE,
                             stdout=sp.PIPE, stderr=sp.PIPE, stdin=sp.PIPE)


        if (resolution is None):
            resolution = [self.metadata["width"], self.metadata["height"], 3]

        self.resolution = resolution

        self.npixels = resolution[0]*resolution[1]*resolution[2]

    def get_metadata(self):
        if self.metadata is None:
            self.read_metadata()

        return self.metadata

    def read_metadata(self):

        meta_command = [ FFPROBE,
                    '-v','quiet',
                    '-print_format', 'json',
                    '-show_format',
                    '-show_streams',
                    self.filename ]
        # note: stdout pipes the data we want
        # stderr captures the status updates ffmpeg would print to the screen
        # stdin prevents ffmpeg from capturing keyboard input.
        meta_pipe = sp.Popen(meta_command, bufsize=BUFFER_SIZE,
                             stdout=sp.PIPE, stderr=sp.PIPE, stdin=sp.PIPE)

        ffprobe_output = meta_pipe.stdout.read(BUFFER_SIZE)
        meta_pipe.close()

        full_metadata = json.loads(ffprobe_output)

        self.metadata = self.get_video_stream(full_metadata)


    def get_video_stream(self, ffprobe_meta):

        for s in ffprobe_meta["streams"]:
            if s["codec_type"] == "video":
                return s

        return None

    def __iter__(self):
        """make the reader object iterable"""
        return self

    def __next__(self):
        try:
            raw_image = self.pipe.stdout.read(self.npixels)
        except:
            # if something breaks here, the pipe may already be closed
            raise StopIteration

        if (len(raw_image) < self.npixels):
            self.close()
            raise StopIteration

        image = np.fromstring(raw_image, dtype='uint8')

        return image.reshape(self.resolution)

    next=__next__

    def close(self):
        """docstring for close"""
        self.pipe.stdout.close()
        self.pipe.stderr.close()
        self.pipe.send_signal(signal.SIGINT)
        self.pipe.wait()


def main():
    """Print purpose of library"""
    print("This is a library for reading video sequences into python via ffmpeg. ")
    print("Provides the 'Video_Reader' iterator class. ")
    print("Requires ffmpeg be installed. ")

if __name__ == '__main__':
    main()
