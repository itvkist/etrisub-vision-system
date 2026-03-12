import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GLib, GstApp

import numpy as np
from threading import Thread, Lock


# ===== GLOBAL MAIN LOOP FOR ALL PIPELINES =====
class _GstMainLoop:
    started = False
    loop = None
    thread = None

    @staticmethod
    def ensure():
        """Start one shared GLib MainLoop for all GStreamer pipelines"""
        if not _GstMainLoop.started:
            #_ = GstApp
            Gst.init()

            _GstMainLoop.loop = GLib.MainLoop()
            _GstMainLoop.thread = Thread(
                target=_GstMainLoop.loop.run, daemon=True
            )
            _GstMainLoop.thread.start()
            _GstMainLoop.started = True


# ===== VIDEO READER CLASS =====
class GStreamerReader:
    def __init__(self, src):
        """
        src: file path or RTSP URL
        """
        _GstMainLoop.ensure()

        self.src = src
        self.lock = Lock()

        # --- Build pipeline ---
        if src.startswith("rtsp://"):
            # RTSP Example
            #pipe = (
                #f"rtspsrc location={src} protocols=tcp user-id=admin user-pw=IT@vkist1 ! "
                #f"rtspsrc location={src} protocols=tcp ! "
                #f"rtph264depay ! h264parse ! nvh264dec ! videoconvert ! "
                #f"video/x-raw,format=BGR ! appsink name=sink"
            #)
            pipe = (f"rtspsrc location={src} protocols=tcp user-id=admin user-pw=IT@vkist1 ! rtph264depay ! h264parse ! nvh264dec ! video/x-raw(memory:CUDAMemory) !"
                    f"videorate ! video/x-raw(memory:CUDAMemory), framerate=9/1 !"
                    f"cudascale ! video/x-raw(memory:CUDAMemory), width=640,height=640 !"
                    f"cudaconvert ! video/x-raw(memory:CUDAMemory) !"
                    f"nvjpegenc quality=80 ! appsink name=sink"
                    #f"cudadownload ! videoconvert ! video/x-raw, format=NV12 ! videoconvert !"
                 #f"jpegenc quality=80 ! appsink name=sink"
                )
            print("Reading from rtsp url")
        else:
            # File Example
            pipe = (
                f"filesrc location={src} ! qtdemux ! h264parse ! nvh264dec ! "
                f"videoconvert ! video/x-raw,format=BGR ! appsink name=sink"
            )
            print("Reading from file source")

        try:
            self.pipeline = Gst.parse_launch(pipe)
        except GLib.Error as e:
            raise RuntimeError(f"GStreamer pipeline parse failed: {e}")

        self.appsink = self.pipeline.get_by_name("sink")
        if self.appsink is None:
            raise RuntimeError("Cannot find appsink!")

        # configure appsink
        self.appsink.set_property("emit-signals", False)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)  # always keep the newest frame

        # run pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Pipeline failed to start")
        
    # ========================
    # MAIN READ() METHOD
    # ========================
    def read_frame(self, timeout_ms=15000):
        """
        Returns: numpy array frame (H, W, 3) or None if timeout / EOS
        """
        with self.lock:  # GIL-safe if multithread
            sample = self.appsink.try_pull_sample(timeout_ms * Gst.MSECOND)
            if not sample:
                print("Failed to read from src")
                return (False, None)

            # get resolution
            caps = sample.get_caps()
            s = caps.get_structure(0)
            width = s.get_value("width")
            height = s.get_value("height")

            # map buffer
            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return (False, None)

            try:
                frame = np.frombuffer(map_info.data, dtype=np.uint8)
                frame = frame.reshape((height, width, 3))
                return (True, frame.copy())  # copy to detach from GStreamer memory
            finally:
                buffer.unmap(map_info)

    def read_jpeg_byted(self, timeout_ms=5000):
        """
        Returns: bytes of encoded frame as jpeg or None if timeout / EOS
        """
        with self.lock:  # GIL-safe if multithread
            sample = self.appsink.try_pull_sample(timeout_ms * Gst.MSECOND)
            if not sample:
                print("Failed to read from src")
                return (False, None)

            try:
                buffer = sample.get_buffer()
                jpeg_bytes = buffer.extract_dup(0, buffer.get_size())
                return (True, jpeg_bytes)  # copy to detach from GStreamer memory
            finally:
                pass

    # ========================
    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)

def worker(path, idx):
    reader = GStreamerReader(path)
    while True:
        _, f = reader.read_jpeg_byted()
        if f is not None:
            print(f"[Cam {idx}] bytes {f}")
        else:
            print(f"[Cam {idx}] timeout")
            break
    reader.stop()

def test():
    #paths = ["LTV_10.mp4", "more_10.mp4"]
    #paths = ["rtsp://172.29.128.1:8554/live/mystream1", "LTV_10.mp4"]
    paths = ["rtsp://localhost:8554/live/mystream1"]
    #paths = ["rtsp://admin:IT@vkist1@10.1.8.224/stream0"]
    #paths = ["rtsp://172.29.128.1:8554/live/mystream1"]
    threads = []

    try:
        for i, p in enumerate(paths):
            t = Thread(target=worker, args=(p, i), daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    except KeyboardInterrupt:
        print("Ctrl+C pressed. Exiting.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    test()