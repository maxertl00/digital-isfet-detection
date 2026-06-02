#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smartloc2_docking.py
====================

A lightweight, dependency-free docking framework for Tkinter.

It provides a dashboard made of four (or more) dockable panels arranged in a
classic IDE-style layout (two columns, each split into rows). Every panel can
be:

  * resized freely by dragging the splitters (sashes) between panels,
  * re-arranged by dragging its title bar onto another panel (with a live
    drop-zone preview highlighting left / right / top / bottom / center),
  * detached into its own floating window ("float") and re-docked again,
  * collapsed/expanded.

The chosen layout (sizes, arrangement and floating windows) is serialised to a
small JSON file so it can be restored on the next launch. A "reset layout"
helper restores the built-in default arrangement.

The framework is intentionally self-contained (only the standard library +
Tkinter) so it can be reused by other SmartLoC2 tools without extra
dependencies.

Author: SmartLoC2 project
"""

from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
#  Theme palette (light scientific theme)                                     #
# --------------------------------------------------------------------------- #

class Palette:
    """Central colour / spacing definitions for the light scientific theme."""

    # Surfaces
    APP_BG = "#EEF1F4"          # window background (cool light grey)
    PANEL_BG = "#FFFFFF"        # panel body
    PANEL_HEADER = "#F3F6F9"    # panel title bar
    PANEL_HEADER_ACTIVE = "#E4ECF3"
    BORDER = "#D2DAE2"          # subtle panel border
    SASH = "#DCE3EA"            # splitter handle
    SASH_ACTIVE = "#2D6E83"     # splitter while dragging

    # Text
    TEXT = "#1F2A33"
    TEXT_MUTED = "#5C6B78"
    TITLE = "#1B4B5A"

    # Accents
    ACCENT = "#2D6E83"          # primary (teal-blue)
    ACCENT_DARK = "#1E4F60"
    GREEN = "#1D6B4B"
    GREEN_DARK = "#134734"
    GOLD = "#C8892B"
    RED = "#B33D34"

    # Status badge colours
    BADGE_IDLE = "#B6C2CC"      # grey  - idle / disconnected
    BADGE_OK = "#2FA36B"        # green - connected / running
    BADGE_WARN = "#C8892B"      # gold  - warning / paused
    BADGE_ERR = "#B33D34"       # red   - error

    # Drop-zone preview
    DROP_FILL = "#2D6E83"

    # Spacing (comfortable density)
    GAP = 10
    SASH_SIZE = 8


# --------------------------------------------------------------------------- #
#  A single dockable panel                                                     #
# --------------------------------------------------------------------------- #

class DockPanel(ttk.Frame):
    """A panel with a title bar, action buttons and a body container.

    The body (``self.body``) is where callers put their own widgets (plots,
    forms, text consoles ...). The panel header exposes float/dock and
    collapse/expand controls and acts as the drag handle for re-arranging.
    """

    def __init__(self, manager: "DockManager", key: str, title: str, icon: str = "▦"):
        super().__init__(manager.host, style="Dock.TFrame")
        self.manager = manager
        self.key = key
        self.title = title
        self.icon_glyph = icon
        self.collapsed = False
        self._float_window: Optional[tk.Toplevel] = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ----- Header / title bar -----
        self.header = ttk.Frame(self, style="DockHeader.TFrame", padding=(10, 6))
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.columnconfigure(2, weight=1)

        # Status badge (coloured dot)
        self.badge = tk.Canvas(self.header, width=12, height=12, highlightthickness=0,
                               bg=Palette.PANEL_HEADER, bd=0)
        self._badge_dot = self.badge.create_oval(2, 2, 11, 11, fill=Palette.BADGE_IDLE,
                                                 outline="")
        self.badge.grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.icon = ttk.Label(self.header, text=icon, style="DockIcon.TLabel")
        self.icon.grid(row=0, column=1, sticky="w", padx=(0, 7))

        self.title_label = ttk.Label(self.header, text=title, style="DockTitle.TLabel")
        self.title_label.grid(row=0, column=2, sticky="w")

        # Action buttons (collapse, float/dock)
        self.btn_collapse = ttk.Label(self.header, text="–", style="DockBtn.TLabel", cursor="hand2")
        self.btn_collapse.grid(row=0, column=3, sticky="e", padx=3)
        self.btn_float = ttk.Label(self.header, text="⧉", style="DockBtn.TLabel", cursor="hand2")
        self.btn_float.grid(row=0, column=4, sticky="e", padx=3)

        self.btn_collapse.bind("<Button-1>", lambda _e: self.toggle_collapsed())
        self.btn_float.bind("<Button-1>", lambda _e: self.toggle_float())

        # ----- Body -----
        self.body = ttk.Frame(self, style="DockBody.TFrame", padding=Palette.GAP)
        self.body.grid(row=1, column=0, sticky="nsew")
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)

        # Make the header a drag handle for re-arranging panels.
        for handle in (self.header, self.title_label, self.icon):
            handle.bind("<ButtonPress-1>", self._on_drag_start)
            handle.bind("<B1-Motion>", self._on_drag_motion)
            handle.bind("<ButtonRelease-1>", self._on_drag_release)
            handle.bind("<Enter>", lambda _e: self._set_header_active(True))
            handle.bind("<Leave>", lambda _e: self._set_header_active(False))

    # --- header visuals ---------------------------------------------------- #
    def _set_header_active(self, active: bool):
        self.header.configure(style="DockHeaderActive.TFrame" if active else "DockHeader.TFrame")

    def set_badge(self, color: str):
        """Update the coloured status dot in the title bar."""
        try:
            self.badge.itemconfigure(self._badge_dot, fill=color)
        except tk.TclError:
            pass

    # --- collapse / expand ------------------------------------------------- #
    def toggle_collapsed(self):
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.body.grid_remove()
            self.btn_collapse.configure(text="+")
        else:
            self.body.grid()
            self.btn_collapse.configure(text="–")
        self.manager.save_layout()

    # --- floating / docking ------------------------------------------------ #
    @property
    def is_floating(self) -> bool:
        return bool(self._float_window)

    def toggle_float(self):
        if self.is_floating:
            self.dock()
        else:
            self.float_out()

    def float_out(self, geometry: Optional[str] = None):
        """Turn this panel into its own top-level window.

        Uses Tk's ``wm manage`` mechanism, which promotes an existing widget to
        a managed top-level *in place* (no reparenting), so the matplotlib
        canvases and all child widgets keep working.
        """
        if self.is_floating:
            return
        # Remove from its paned window first so the slot collapses cleanly.
        self.manager._detach_panel(self)
        try:
            self.tk.call("wm", "manage", self._w)
            self.tk.call("wm", "title", self._w, self.title)
            self.tk.call("wm", "protocol", self._w, "WM_DELETE_WINDOW",
                         self.register(self.dock))
            if geometry:
                self.tk.call("wm", "geometry", self._w, geometry)
            else:
                self.tk.call("wm", "geometry", self._w, "640x460")
        except tk.TclError:
            # If wm manage is unavailable, abort floating and re-dock.
            self.manager._reattach_panel(self)
            return
        self._float_window = True  # sentinel: managed as top-level
        self.btn_float.configure(text="⤢")  # "dock back" glyph
        self.manager.save_layout()

    def dock(self):
        if not self.is_floating:
            return
        try:
            self.tk.call("wm", "forget", self._w)
        except tk.TclError:
            pass
        self._float_window = None
        self.btn_float.configure(text="⧉")
        self.manager._reattach_panel(self)
        self.manager.save_layout()

    def float_geometry(self) -> Optional[str]:
        """Return the current geometry string while floating, else ``None``."""
        if not self.is_floating:
            return None
        try:
            return self.tk.call("wm", "geometry", self._w)
        except tk.TclError:
            return None

    # --- drag to re-arrange ------------------------------------------------ #
    def _on_drag_start(self, event):
        if self.is_floating:
            return
        self.manager.begin_drag(self, event.x_root, event.y_root)

    def _on_drag_motion(self, event):
        if self.is_floating:
            return
        self.manager.update_drag(event.x_root, event.y_root)

    def _on_drag_release(self, _event):
        if self.is_floating:
            return
        self.manager.end_drag()


# --------------------------------------------------------------------------- #
#  Scrollable frame helper                                                     #
# --------------------------------------------------------------------------- #

class ScrollableFrame(ttk.Frame):
    """A vertically scrollable container.

    Put your widgets inside ``self.inner``. A scrollbar appears automatically
    and the mouse wheel scrolls when the pointer is over the area. This keeps
    tall panels (e.g. the configuration form) fully reachable no matter how
    small the user makes the panel.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0, bg=Palette.PANEL_BG, bd=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.inner = ttk.Frame(self.canvas, style="DockBody.TFrame")
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse-wheel scrolling (cross-platform).
        for widget in (self.canvas, self.inner):
            widget.bind("<Enter>", lambda _e: self._bind_wheel())
            widget.bind("<Leave>", lambda _e: self._unbind_wheel())

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # keep the inner frame as wide as the canvas
        self.canvas.itemconfigure(self._win, width=event.width)

    def _bind_wheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta, "units")


# --------------------------------------------------------------------------- #
#  Drop-zone preview overlay                                                   #
# --------------------------------------------------------------------------- #

class DropOverlay:
    """Semi-transparent highlight showing where a dragged panel will land."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.win: Optional[tk.Toplevel] = None

    def show(self, x: int, y: int, w: int, h: int):
        if self.win is None:
            self.win = tk.Toplevel(self.root)
            self.win.overrideredirect(True)
            try:
                self.win.attributes("-alpha", 0.35)
            except tk.TclError:
                pass
            self.win.configure(bg=Palette.DROP_FILL)
            self.win.lift()
        self.win.geometry(f"{max(w, 1)}x{max(h, 1)}+{x}+{y}")
        try:
            self.win.deiconify()
            self.win.lift()
        except tk.TclError:
            pass

    def hide(self):
        if self.win is not None:
            try:
                self.win.withdraw()
            except tk.TclError:
                pass


# --------------------------------------------------------------------------- #
#  The dock manager                                                            #
# --------------------------------------------------------------------------- #

# A "slot" identifies a position in the 2-column / 2-row dashboard grid.
#   left-top, left-bottom, right-top, right-bottom
DEFAULT_SLOTS = ("left-top", "left-bottom", "right-top", "right-bottom")


class DockManager:
    """Builds and manages the dockable dashboard.

    Layout model
    ------------
    The dashboard is a horizontal :class:`ttk.PanedWindow` (left | right). Each
    side is a vertical :class:`ttk.PanedWindow` (top / bottom). Panels are
    assigned to one of four slots. Splitters between panels are the native sash
    handles of the paned windows, so they are freely draggable. Panels can also
    be floated into their own windows and re-docked.
    """

    def __init__(self, root: tk.Tk, host: ttk.Frame, layout_path: str):
        self.root = root
        self.host = host
        self.layout_path = layout_path

        self.panels: Dict[str, DockPanel] = {}
        self.slot_of: Dict[str, str] = {}            # panel key -> slot
        self.default_slot: Dict[str, str] = {}       # panel key -> default slot
        self.overlay = DropOverlay(root)

        # drag state
        self._drag_panel: Optional[DockPanel] = None
        self._drag_target: Optional[Tuple[str, str]] = None  # (slot, zone)

        # ----- Paned window scaffolding -----
        self.host.columnconfigure(0, weight=1)
        self.host.rowconfigure(0, weight=1)

        self.h_paned = ttk.PanedWindow(host, orient="horizontal", style="Dock.TPanedwindow")
        self.h_paned.grid(row=0, column=0, sticky="nsew")

        self.left_paned = ttk.PanedWindow(self.h_paned, orient="vertical", style="Dock.TPanedwindow")
        self.right_paned = ttk.PanedWindow(self.h_paned, orient="vertical", style="Dock.TPanedwindow")
        self.h_paned.add(self.left_paned, weight=2)
        self.h_paned.add(self.right_paned, weight=3)

        # Slot -> containing paned window
        self._paned_of_slot = {
            "left-top": self.left_paned,
            "left-bottom": self.left_paned,
            "right-top": self.right_paned,
            "right-bottom": self.right_paned,
        }

    # ----- registration ---------------------------------------------------- #
    def add_panel(self, key: str, title: str, default_slot: str, icon: str = "▦") -> DockPanel:
        panel = DockPanel(self, key, title, icon=icon)
        self.panels[key] = panel
        self.default_slot[key] = default_slot
        self.slot_of[key] = default_slot
        return panel

    def finalize(self):
        """Place all registered panels in their slots and apply saved layout."""
        self._rebuild_panes()
        self.load_layout()

    # ----- pane (re)building ---------------------------------------------- #
    def _ordered_panels_for(self, paned: ttk.PanedWindow, top_slot: str, bottom_slot: str) -> List[DockPanel]:
        items = []
        for key, slot in self.slot_of.items():
            panel = self.panels[key]
            if panel.is_floating:
                continue
            if slot == top_slot:
                items.append((0, panel))
            elif slot == bottom_slot:
                items.append((1, panel))
        items.sort(key=lambda t: t[0])
        return [p for _, p in items]

    def _rebuild_panes(self):
        """Re-add docked panels to their paned windows in the correct order."""
        for paned, top, bottom in (
            (self.left_paned, "left-top", "left-bottom"),
            (self.right_paned, "right-top", "right-bottom"),
        ):
            # forget everything currently managed
            for child in paned.panes():
                paned.forget(child)
            for panel in self._ordered_panels_for(paned, top, bottom):
                paned.add(panel, weight=1)

    def _detach_panel(self, panel: DockPanel):
        """Remove a panel from its paned window (used before floating)."""
        paned = self._paned_of_slot[self.slot_of[panel.key]]
        if str(panel) in [str(p) for p in paned.panes()]:
            paned.forget(panel)

    def _reattach_panel(self, panel: DockPanel):
        """Re-insert a previously floating panel into its slot."""
        self._rebuild_panes()

    # ----- drag & drop ----------------------------------------------------- #
    def begin_drag(self, panel: DockPanel, x_root: int, y_root: int):
        self._drag_panel = panel
        self._drag_target = None
        self.update_drag(x_root, y_root)

    def _slot_under_pointer(self, x_root: int, y_root: int) -> Optional[str]:
        for slot in DEFAULT_SLOTS:
            panel = self._panel_in_slot(slot)
            if panel is None or panel.is_floating:
                continue
            try:
                px, py = panel.winfo_rootx(), panel.winfo_rooty()
                pw, ph = panel.winfo_width(), panel.winfo_height()
            except tk.TclError:
                continue
            if px <= x_root <= px + pw and py <= y_root <= py + ph:
                return slot
        return None

    def _panel_in_slot(self, slot: str) -> Optional[DockPanel]:
        for key, s in self.slot_of.items():
            if s == slot and not self.panels[key].is_floating:
                return self.panels[key]
        return None

    @staticmethod
    def _zone_in_panel(panel: DockPanel, x_root: int, y_root: int) -> str:
        px, py = panel.winfo_rootx(), panel.winfo_rooty()
        pw, ph = panel.winfo_width(), panel.winfo_height()
        rx = (x_root - px) / max(pw, 1)
        ry = (y_root - py) / max(ph, 1)
        edge = 0.28
        if rx < edge:
            return "left"
        if rx > 1 - edge:
            return "right"
        if ry < edge:
            return "top"
        if ry > 1 - edge:
            return "bottom"
        return "center"

    def update_drag(self, x_root: int, y_root: int):
        if self._drag_panel is None:
            return
        slot = self._slot_under_pointer(x_root, y_root)
        if slot is None:
            self.overlay.hide()
            self._drag_target = None
            return
        target_panel = self._panel_in_slot(slot)
        if target_panel is None:
            self.overlay.hide()
            self._drag_target = None
            return
        zone = self._zone_in_panel(target_panel, x_root, y_root)
        self._drag_target = (slot, zone)

        px, py = target_panel.winfo_rootx(), target_panel.winfo_rooty()
        pw, ph = target_panel.winfo_width(), target_panel.winfo_height()
        if zone == "left":
            self.overlay.show(px, py, pw // 2, ph)
        elif zone == "right":
            self.overlay.show(px + pw // 2, py, pw // 2, ph)
        elif zone == "top":
            self.overlay.show(px, py, pw, ph // 2)
        elif zone == "bottom":
            self.overlay.show(px, py + ph // 2, pw, ph // 2)
        else:
            self.overlay.show(px, py, pw, ph)

    def end_drag(self):
        self.overlay.hide()
        panel = self._drag_panel
        target = self._drag_target
        self._drag_panel = None
        self._drag_target = None
        if panel is None or target is None:
            return
        target_slot, zone = target
        if zone == "center":
            # swap the two panels' slots
            other = self._panel_in_slot(target_slot)
            if other is not None and other is not panel:
                self.slot_of[panel.key], self.slot_of[other.key] = (
                    self.slot_of[other.key],
                    self.slot_of[panel.key],
                )
        else:
            # move panel into the requested half of the target slot's column
            self.slot_of[panel.key] = self._resolve_zone_slot(target_slot, zone)
        self._rebuild_panes()
        self.save_layout()

    @staticmethod
    def _resolve_zone_slot(target_slot: str, zone: str) -> str:
        side = "left" if target_slot.startswith("left") else "right"
        if zone in ("left",):
            side = "left"
        elif zone in ("right",):
            side = "right"
        row = "top"
        if zone == "bottom":
            row = "bottom"
        elif zone == "top":
            row = "top"
        else:
            # left/right zone keeps the row of the target slot
            row = "bottom" if target_slot.endswith("bottom") else "top"
        return f"{side}-{row}"

    # ----- layout persistence --------------------------------------------- #
    def reset_layout(self):
        # dock any floating panels
        for panel in list(self.panels.values()):
            if panel.is_floating:
                panel.dock()
            panel.collapsed = False
            panel.body.grid()
            panel.btn_collapse.configure(text="–")
        self.slot_of = dict(self.default_slot)
        self._rebuild_panes()
        self.root.update_idletasks()
        self._apply_default_sashes()
        self.save_layout()

    def _apply_default_sashes(self):
        try:
            self.h_paned.update_idletasks()
            total_w = self.h_paned.winfo_width()
            if total_w > 1:
                self.h_paned.sashpos(0, int(total_w * 0.34))
            for paned in (self.left_paned, self.right_paned):
                paned.update_idletasks()
                total_h = paned.winfo_height()
                if total_h > 1:
                    paned.sashpos(0, int(total_h * 0.55))
        except tk.TclError:
            pass

    def save_layout(self):
        try:
            state = {
                "slots": dict(self.slot_of),
                "collapsed": {k: p.collapsed for k, p in self.panels.items()},
                "floating": {},
                "sashes": {},
            }
            for key, panel in self.panels.items():
                if panel.is_floating:
                    geom = panel.float_geometry()
                    if geom:
                        state["floating"][key] = geom
            try:
                state["sashes"]["h"] = self.h_paned.sashpos(0)
                state["sashes"]["left"] = self.left_paned.sashpos(0)
                state["sashes"]["right"] = self.right_paned.sashpos(0)
            except tk.TclError:
                pass
            with open(self.layout_path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except OSError:
            pass

    def load_layout(self):
        if not os.path.isfile(self.layout_path):
            self.root.after(120, self._apply_default_sashes)
            return
        try:
            with open(self.layout_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, ValueError):
            self.root.after(120, self._apply_default_sashes)
            return

        slots = state.get("slots", {})
        for key, slot in slots.items():
            if key in self.slot_of and slot in DEFAULT_SLOTS:
                self.slot_of[key] = slot

        self._rebuild_panes()

        # collapsed states
        for key, collapsed in state.get("collapsed", {}).items():
            panel = self.panels.get(key)
            if panel is not None and collapsed:
                panel.toggle_collapsed()

        # floating windows
        for key, geom in state.get("floating", {}).items():
            panel = self.panels.get(key)
            if panel is not None:
                panel.float_out(geometry=geom)

        # restore sash positions after the window has settled
        sashes = state.get("sashes", {})

        def _restore_sashes():
            try:
                if "h" in sashes:
                    self.h_paned.sashpos(0, int(sashes["h"]))
                if "left" in sashes:
                    self.left_paned.sashpos(0, int(sashes["left"]))
                if "right" in sashes:
                    self.right_paned.sashpos(0, int(sashes["right"]))
            except (tk.TclError, ValueError, TypeError):
                pass

        self.root.after(150, _restore_sashes)


# --------------------------------------------------------------------------- #
#  Styling helper                                                              #
# --------------------------------------------------------------------------- #

def apply_light_scientific_theme(root: tk.Tk, base_font: Tuple[str, int]) -> ttk.Style:
    """Configure ttk styles for the light scientific theme + dock widgets."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    p = Palette
    root.configure(bg=p.APP_BG)

    family, size = base_font

    # Generic
    style.configure(".", font=base_font, background=p.APP_BG, foreground=p.TEXT)
    style.configure("TFrame", background=p.APP_BG)
    style.configure("TLabel", background=p.APP_BG, foreground=p.TEXT, padding=(2, 2))
    style.configure("TEntry", padding=(7, 6), fieldbackground="#FFFFFF")
    style.configure("TButton", padding=(11, 7))
    style.map("TButton",
              foreground=[("disabled", "#A9B4BD")],
              background=[("active", "#E4ECF3")])
    style.configure("TCheckbutton", background=p.PANEL_BG, foreground=p.TEXT, padding=(2, 2))
    style.configure("TRadiobutton", background=p.PANEL_BG, foreground=p.TEXT, padding=(2, 2))
    style.configure("TCombobox", padding=4)
    style.configure("TLabelframe", background=p.PANEL_BG, bordercolor=p.BORDER,
                    relief="solid", borderwidth=1, padding=10)
    style.configure("TLabelframe.Label", background=p.PANEL_BG, foreground=p.TITLE,
                    font=(family, size, "bold"), padding=(4, 0))

    # Paned windows
    style.configure("Dock.TPanedwindow", background=p.SASH)
    style.configure("Dock.TPanedwindow.Sash", sashthickness=p.SASH_SIZE,
                    gripcount=0, background=p.SASH)
    style.map("Dock.TPanedwindow.Sash", background=[("active", p.SASH_ACTIVE)])

    # Dock panels
    style.configure("Dock.TFrame", background=p.PANEL_BG, bordercolor=p.BORDER,
                    relief="solid", borderwidth=1)
    style.configure("DockBody.TFrame", background=p.PANEL_BG)
    style.configure("DockHeader.TFrame", background=p.PANEL_HEADER)
    style.configure("DockHeaderActive.TFrame", background=p.PANEL_HEADER_ACTIVE)
    style.configure("DockTitle.TLabel", background=p.PANEL_HEADER, foreground=p.TITLE,
                    font=(family, size, "bold"))
    style.configure("DockIcon.TLabel", background=p.PANEL_HEADER, foreground=p.ACCENT,
                    font=(family, size + 1, "bold"))
    style.configure("DockBtn.TLabel", background=p.PANEL_HEADER, foreground=p.TEXT_MUTED,
                    font=(family, size + 1, "bold"))

    # Inner section frames inside panels
    style.configure("Section.TLabelframe", background=p.PANEL_BG, bordercolor=p.BORDER,
                    relief="solid", borderwidth=1, padding=10)
    style.configure("Section.TLabelframe.Label", background=p.PANEL_BG,
                    foreground=p.ACCENT_DARK, font=(family, size, "bold"))
    style.configure("Status.TLabel", background=p.PANEL_BG, foreground=p.ACCENT_DARK)
    style.configure("Muted.TLabel", background=p.PANEL_BG, foreground=p.TEXT_MUTED)

    return style
