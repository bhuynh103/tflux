import pyglet
from pyglet.gl import (
    glEnable, glDisable, glPointSize, glClearColor,
    glMatrixMode, glLoadIdentity, glRotatef, glTranslatef, glScalef,
    GL_DEPTH_TEST, GL_PROJECTION, GL_MODELVIEW,
)
from pyglet import math as pmath
from tflux.dtypes import Junction
import numpy as np
from logging import getLogger

logger = getLogger(__name__)


# Distinct colours per roi_index (RGB 0-255)
_ROI_COLOURS = [
    (220,  60,  60),   # 0 — red
    ( 60, 180,  60),   # 1 — green
    ( 60, 100, 220),   # 2 — blue
    (220, 160,  40),   # 3 — amber
    (160,  60, 220),   # 4 — violet
]


def render_junction(junction: Junction, point_size: float = 3.0, window_size: int = 800) -> None:
    """
    Opens a pyglet window and renders a Junction as a 3-D point cloud.

    Controls:
      Left-drag   — rotate
      Scroll      — zoom
      R           — reset view
      Q / Escape  — close

    Args:
        junction:    A Junction object with a .vertices (N,3) array in (t,y,x) order
                     and a .roi_index int.
        point_size:  GL point size in pixels.
        window_size: Square window width/height in pixels.
    """
    verts = np.asarray(junction.vertices, dtype=np.float32)   # (N, 3) — t, y, x
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected vertices shape (N,3), got {verts.shape}")

    N = verts.shape[0]
    logger.info(f"Rendering Junction roi_index={junction.roi_index} with {N} vertices")

    # Re-order to (x, y, t) for OpenGL axes
    xyz = verts[:, [2, 1, 0]].copy()   # x, y, t → OpenGL X, Y, Z

    # Centre and normalise to [-1, 1]
    centre = xyz.mean(axis=0)
    xyz -= centre
    scale = np.abs(xyz).max() + 1e-12
    xyz /= scale

    colour_rgb = _ROI_COLOURS[junction.roi_index % len(_ROI_COLOURS)]
    flat_verts = xyz.flatten().tolist()
    flat_colours = (colour_rgb * N)   # repeat colour N times

    # ---- pyglet setup ----
    window = pyglet.window.Window(
        width=window_size,
        height=window_size,
        caption=f"Junction roi_index={junction.roi_index}  |  {N} pts",
        resizable=True,
    )

    batch = pyglet.graphics.Batch()
    point_list = batch.add(
        N,
        pyglet.gl.GL_POINTS,
        None,
        ("v3f/static", flat_verts),
        ("c3B/static", flat_colours),
    )

    # Mutable view state
    state = {"rot_x": 20.0, "rot_y": -30.0, "zoom": 1.0,
             "drag_x": 0, "drag_y": 0, "dragging": False}

    @window.event
    def on_resize(width, height):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / max(height, 1)
        pyglet.gl.gluPerspective(45.0, aspect, 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED

    @window.event
    def on_draw():
        window.clear()
        glEnable(GL_DEPTH_TEST)
        glPointSize(point_size)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3.0)
        glScalef(state["zoom"], state["zoom"], state["zoom"])
        glRotatef(state["rot_x"], 1.0, 0.0, 0.0)
        glRotatef(state["rot_y"], 0.0, 1.0, 0.0)
        batch.draw()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            state["dragging"] = True
            state["drag_x"] = x
            state["drag_y"] = y

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            state["dragging"] = False

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if state["dragging"]:
            state["rot_y"] += dx * 0.5
            state["rot_x"] -= dy * 0.5

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        state["zoom"] = max(0.1, state["zoom"] + scroll_y * 0.05)

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol in (pyglet.window.key.Q, pyglet.window.key.ESCAPE):
            window.close()
        elif symbol == pyglet.window.key.R:
            state["rot_x"] = 20.0
            state["rot_y"] = -30.0
            state["zoom"] = 1.0

    glClearColor(0.12, 0.12, 0.14, 1.0)
    on_resize(window_size, window_size)
    pyglet.app.run()