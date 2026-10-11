"""Tkinter player: a worker thread reads frames and runs the model, the Tk thread draws them."""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from harness.config import HarnessConfig, load_config
from harness.draw import draw_pose
from harness.pose import Pose, PoseEstimator
from harness.source import Source, open_source

POLL_MS = 10

# TensorFlow's C++ info/warning logs; set before TensorFlow is first imported (lazily, in pose.py).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@contextlib.contextmanager
def quiet_stderr():
    """Send file descriptor 2 to /dev/null while TensorFlow initializes.

    absl writes a few start-up lines straight to stderr regardless of
    TF_CPP_MIN_LOG_LEVEL. stderr is restored before any exception propagates,
    so a failed load still prints its traceback.
    """
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class Worker(threading.Thread):
    """Reads frames at the source's frame rate and estimates a pose for each one.

    Results go to a one-slot queue that always holds the newest frame. If inference
    is slower than real time, a video plays slower rather than skipping frames.
    """

    def __init__(self, cfg: HarnessConfig, estimator: PoseEstimator, source: Source):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.estimator = estimator
        self.source = source
        self.frames: queue.Queue[tuple[np.ndarray, Pose, float]] = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.paused = threading.Event()
        self.restart = threading.Event()
        self.ended = False
        self.error: str | None = None

    def run(self) -> None:
        try:
            self._loop()
        except Exception as e:  # surfaced in the status bar
            self.error = f"{type(e).__name__}: {e}"

    def _loop(self) -> None:
        cap, interval = self.source.cap, 1.0 / self.source.fps
        prev: Pose | None = None
        next_time = time.perf_counter()
        while not self.stop.is_set():
            if self.restart.is_set():
                self.restart.clear()
                if not self.source.is_live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                prev, self.ended = None, False
                self.paused.clear()
            if self.paused.is_set():
                self.stop.wait(0.05)
                next_time = time.perf_counter()
                continue

            ok, bgr = cap.read()
            if not ok:
                if self.source.is_live:
                    raise RuntimeError("webcam stopped delivering frames")
                if self.cfg.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    prev = None
                else:
                    self.ended = True
                    self.paused.set()
                continue

            t0 = time.perf_counter()
            pose = self.estimator.predict(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), prev if self.cfg.track else None
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0
            prev = pose

            if not self.source.is_live:
                next_time += interval
                delay = next_time - time.perf_counter()
                if delay > 0:
                    self.stop.wait(delay)
                elif delay < -0.25:  # fell behind: don't burst to catch up
                    next_time = time.perf_counter()

            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put((bgr, pose, infer_ms))


class App:
    def __init__(self, root: tk.Tk, cfg: HarnessConfig, estimator: PoseEstimator, source: Source):
        self.root, self.cfg, self.source, self.estimator = root, cfg, source, estimator
        self.worker = Worker(cfg, estimator, source)
        self.show_skeleton = tk.BooleanVar(value=True)
        self.last: tuple[np.ndarray, Pose, float] | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.shown, self.fps_clock, self.display_fps = 0, time.perf_counter(), 0.0

        root.title(f"Pose harness: {source.label}")
        root.geometry(f"{cfg.window_width}x{cfg.window_height}")
        root.configure(background="black")

        self.area = tk.Frame(root, background="black")
        self.area.pack(fill=tk.BOTH, expand=True)
        self.view = tk.Label(self.area, background="black", foreground="white", text="Loading…")
        # place() rather than pack() so the image never resizes the window.
        self.view.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        bar = ttk.Frame(root, padding=4)
        bar.pack(fill=tk.X)
        self.play_btn = ttk.Button(bar, text="Pause", width=8, command=self.toggle_pause, takefocus=False)
        self.play_btn.pack(side=tk.LEFT)
        ttk.Button(bar, text="Restart", command=self.worker.restart.set, takefocus=False).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(
            bar, text="Skeleton", variable=self.show_skeleton, command=self.redraw, takefocus=False
        ).pack(side=tk.LEFT, padx=4)
        self.status = ttk.Label(bar, text="")
        self.status.pack(side=tk.LEFT, padx=8)

        root.bind("<space>", lambda _: self.toggle_pause())
        root.bind("<r>", lambda _: self.worker.restart.set())
        root.bind("<s>", lambda _: (self.show_skeleton.set(not self.show_skeleton.get()), self.redraw()))
        self.area.bind("<Configure>", lambda _: self.redraw())
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.worker.start()
        root.after(POLL_MS, self.poll)

    def toggle_pause(self) -> None:
        if self.worker.ended:
            self.worker.restart.set()
        elif self.worker.paused.is_set():
            self.worker.paused.clear()
        else:
            self.worker.paused.set()

    def poll(self) -> None:
        try:
            self.last = self.worker.frames.get_nowait()
        except queue.Empty:
            pass
        else:
            self.redraw()
            self.shown += 1
        now = time.perf_counter()
        if now - self.fps_clock >= 1.0:
            self.display_fps = self.shown / (now - self.fps_clock)
            self.shown, self.fps_clock = 0, now
        self.update_status()
        self.root.after(POLL_MS, self.poll)

    def redraw(self) -> None:
        """Overlay the pose on the full-resolution frame, then scale it to fit the window."""
        if self.last is None:
            return
        bgr, pose, _ = self.last
        if self.show_skeleton.get():
            bgr = draw_pose(bgr, pose, self.cfg.score_threshold)
        aw, ah = self.area.winfo_width(), self.area.winfo_height()
        if aw < 2 or ah < 2:
            return
        h, w = bgr.shape[:2]
        k = min(aw / w, ah / h)
        size = (max(1, round(w * k)), max(1, round(h * k)))
        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
        rgb = cv2.cvtColor(cv2.resize(bgr, size, interpolation=interp), cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.view.configure(image=photo, text="")
        self.photo = photo  # Tk doesn't hold a reference

    def update_status(self) -> None:
        w = self.worker
        if w.error:
            text = f"error: {w.error}"
        else:
            parts = [self.source.label, self.estimator.backend.label]
            if self.last is not None:
                h, wd = self.last[0].shape[:2]
                parts += [
                    f"{wd}x{h} @ {self.source.fps:.0f} fps",
                    f"inference {self.last[2]:.1f} ms",
                    f"pose score {self.last[1].score:.2f}",
                    f"display {self.display_fps:.1f} fps",
                ]
            if w.ended:
                parts.append("ended")
            elif w.paused.is_set():
                parts.append("paused")
            text = "  |  ".join(parts)
        self.status.configure(text=text)
        self.play_btn.configure(text="Play" if w.paused.is_set() else "Pause")

    def close(self) -> None:
        self.worker.stop.set()
        self.worker.join(timeout=2.0)
        self.source.cap.release()
        self.root.destroy()


def run(cfg: HarnessConfig) -> None:
    print(f"loading model {cfg.model_path}")
    with quiet_stderr():
        estimator = PoseEstimator(
            cfg.model_path, cfg.flip_test, cfg.score_threshold, cfg.num_threads, cfg.heatmap_coords
        )
    root = tk.Tk()
    root.withdraw()  # stay hidden while the file picker (if any) is open
    source = open_source(cfg, parent=root)
    if source is None:
        root.destroy()
        return
    App(root, cfg, estimator, source)
    root.deiconify()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose estimation overlay on webcam or video.")
    parser.add_argument("--config", default="config.json", help="path to the harness JSON config")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
