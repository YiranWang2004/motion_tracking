"""Tk playback controls, serviced by the viewer's main thread."""

from __future__ import annotations

from omnicontact.replay import ReplayClock


class ReplayControls:
    def __init__(self, clock: ReplayClock) -> None:
        import tkinter as tk

        self.clock = clock
        self.dragging = False
        self.resume_after_drag = False
        self.closed = False
        self.root = tk.Tk()
        self.root.title("Dual ScaleBFM replay")
        self.root.geometry("760x150")
        self.root.minsize(420, 150)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.position = tk.IntVar(master=self.root, value=clock.index)
        self.slider = tk.Scale(
            self.root, from_=0, to=clock.replay.frame_count - 1,
            orient=tk.HORIZONTAL, resolution=1, showvalue=False,
            variable=self.position, command=self._seek,
        )
        self.slider.pack(fill="x", padx=16, pady=(10, 0))
        self.slider.bind("<ButtonPress-1>", self._begin_drag)
        self.slider.bind("<ButtonRelease-1>", self._end_drag)
        self.status = tk.Label(self.root)
        self.status.pack()
        self.play = tk.Button(self.root, command=clock.toggle_pause)
        self.play.pack(pady=6)

    def _begin_drag(self, _event=None) -> None:
        self.dragging = True
        self.resume_after_drag = not self.clock.paused
        self.clock.seek(self.clock.index, paused=True)

    def _seek(self, value: str) -> None:
        # IntVar updates used for playback tracking must never seek the clock.
        if self.dragging:
            self.clock.seek(int(float(value)), paused=True)

    def _end_drag(self, _event=None) -> None:
        if self.dragging:
            self.clock.seek(self.position.get(), paused=not self.resume_after_drag)
            self.dragging = False

    def update(self) -> None:
        if self.closed:
            return
        self.root.update()
        if self.closed:
            return
        clock = self.clock
        if not self.dragging:
            self.position.set(clock.index)
        self.status.configure(text=(
            f"Frame {clock.index} / {clock.replay.frame_count - 1}    "
            f"{clock.replay.elapsed_s[clock.index]:.3f} / "
            f"{clock.replay.duration_s:.3f} s    {clock.speed:g}x"
        ))
        self.play.configure(text="Play" if clock.paused else "Pause")

    def close(self) -> None:
        if not self.closed:
            self._end_drag()
            self.closed = True
            self.root.destroy()
