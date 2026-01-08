from __future__ import annotations

import argparse
import ctypes
import json
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Literal, Optional

from PIL import Image, ImageTk

from .capture import ScreenCapturer


ItemKind = Literal["roi", "anchor"]


@dataclass
class RoiDraft:
    name: str = ""
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    description: str = ""
    tags: str = ""
    active: bool = True

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "description": self.description,
            "active": bool(self.active),
        }
        tags = [t.strip() for t in (self.tags or "").split(",") if t.strip()]
        if tags:
            out["tags"] = tags
        return out


@dataclass
class AnchorDraft:
    name: str = ""
    x: int = 0
    y: int = 0
    description: str = ""
    tags: str = ""
    linked_rois: list[str] = field(default_factory=list)
    active: bool = True

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "x": int(self.x),
            "y": int(self.y),
            "description": self.description,
            "active": bool(self.active),
        }
        tags = [t.strip() for t in (self.tags or "").split(",") if t.strip()]
        if tags:
            out["tags"] = tags
        linked = [str(x) for x in (self.linked_rois or []) if str(x).strip()]
        if linked:
            out["linked_ROIs"] = linked
        return out


def _load_workspace_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rois": [], "anchors": [], "tools": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("workspace must be a JSON object")
    raw.setdefault("rois", [])
    raw.setdefault("anchors", [])
    raw.setdefault("tools", {})
    if not isinstance(raw["rois"], list) or not isinstance(raw["anchors"], list) or not isinstance(raw["tools"], dict):
        raise ValueError("workspace fields must be {rois:list, anchors:list, tools:object}")
    return raw


def _safe_int(value: str, *, field: str) -> int:
    try:
        return int(float(value.strip()))
    except Exception:
        raise ValueError(f"Invalid integer for {field}: {value!r}")


def _dedupe_name(existing: set[str], base: str) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _parse_active(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return True


def _enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
    except Exception:
        pass


class CalibratorApp(tk.Tk):
    def __init__(self, *, workspace_path: Path) -> None:
        super().__init__()
        self.title("Workspace Calibrator (ROIs + Anchors)")
        self.geometry("1400x900")
        self._apply_tk_scaling()

        self.workspace_path = workspace_path
        self.raw = _load_workspace_raw(workspace_path)

        self.rois: list[RoiDraft] = []
        for r in self.raw.get("rois", []):
            if isinstance(r, dict):
                self.rois.append(
                    RoiDraft(
                        name=str(r.get("name", "")),
                        x=int(r.get("x", 0)),
                        y=int(r.get("y", 0)),
                        w=int(r.get("w", 0)),
                        h=int(r.get("h", 0)),
                        description=str(r.get("description", "")),
                        tags=",".join(str(t) for t in (r.get("tags") or []) if isinstance(t, (str, int, float))),
                        active=_parse_active(r.get("active", True)),
                    )
                )

        self.anchors: list[AnchorDraft] = []
        for a in self.raw.get("anchors", []):
            if isinstance(a, dict):
                linked = a.get("linked_ROIs", []) or []
                linked_list = [str(x) for x in linked if isinstance(x, (str, int, float)) and str(x).strip()] if isinstance(linked, list) else []
                self.anchors.append(
                    AnchorDraft(
                        name=str(a.get("name", "")),
                        x=int(a.get("x", 0)),
                        y=int(a.get("y", 0)),
                        description=str(a.get("description", "")),
                        tags=",".join(str(t) for t in (a.get("tags") or []) if isinstance(t, (str, int, float))),
                        linked_rois=linked_list,
                        active=_parse_active(a.get("active", True)),
                    )
                )

        self.mode: Literal["idle", "draw_roi", "pick_anchor"] = "idle"
        self._drawing_item_index: Optional[int] = None
        self._draw_start: Optional[tuple[int, int]] = None
        self._active_rect_id: Optional[int] = None
        self._active_crosshair: list[int] = []

        self._screen_img: Optional[Image.Image] = None
        self._screen_photo: Optional[ImageTk.PhotoImage] = None
        self._screen_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # left, top, width, height
        self._scale: float = 1.0
        self._fit_scale: float = 1.0
        self._zoom: float = 1.0
        self._render_job: Optional[str] = None
        self._suppress_form_events: bool = False
        self._help_window: Optional[tk.Toplevel] = None
        self._help_btn: Optional[ttk.Button] = None
        self._desc_text: Optional[tk.Text] = None
        self._form_canvas: Optional[tk.Canvas] = None
        self._form_inner: Optional[ttk.Frame] = None
        self._form_window: Optional[int] = None

        self._build_ui()
        self._refresh_screenshot()
        self._refresh_list()

    def _apply_tk_scaling(self) -> None:
        # Align Tk scaling to the system DPI so text/widgets scale with Windows Display Scaling.
        try:
            dpi = float(self.winfo_fpixels("1i"))
            self.tk.call("tk", "scaling", dpi / 72.0)
        except Exception:
            pass

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        paned = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=6,
            sashpad=2,
            showhandle=True,
            handlesize=8,
            sashcursor="sb_h_double_arrow",
        )
        paned.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(paned, padding=8)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Items").grid(row=0, column=0, sticky="w")
        items_frame = ttk.Frame(left)
        items_frame.grid(row=1, column=0, sticky="nsew")
        items_frame.columnconfigure(0, weight=1)
        items_frame.rowconfigure(0, weight=1)

        self.items_list = tk.Listbox(items_frame, width=36, height=14, exportselection=False)
        items_scroll = ttk.Scrollbar(items_frame, orient="vertical", command=self.items_list.yview)
        self.items_list.configure(yscrollcommand=items_scroll.set)
        self.items_list.grid(row=0, column=0, sticky="nsew")
        items_scroll.grid(row=0, column=1, sticky="ns")
        self.items_list.bind("<<ListboxSelect>>", lambda _e: self._on_select(populate_form=True))
        self.items_list.bind("<Button-1>", self._on_items_click, add="+")
        self._items_default_fg = str(self.items_list.cget("fg"))

        btns = ttk.Frame(left)
        btns.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        ttk.Button(btns, text="Add ROI", command=self._add_roi).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(btns, text="Add Anchor", command=self._add_anchor).grid(row=0, column=1, sticky="ew")
        ttk.Button(btns, text="Delete", command=self._delete_selected).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(btns, text="Save", command=self._save).grid(row=1, column=1, sticky="ew", pady=(6, 0))

        tools = ttk.LabelFrame(left, text="Pick on screenshot", padding=8)
        tools.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(tools, text="Draw ROI box", command=self._begin_draw_roi).grid(row=0, column=0, sticky="ew")
        ttk.Button(tools, text="Pick anchor point", command=self._begin_pick_anchor).grid(
            row=1, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Button(tools, text="Refresh screenshot", command=self._refresh_screenshot).grid(
            row=2, column=0, sticky="ew", pady=(6, 0)
        )

        right = ttk.Frame(paned, padding=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        paned.add(left)
        paned.add(right)
        try:
            paned.paneconfigure(left, minsize=260, stretch="never")
            paned.paneconfigure(right, minsize=600, stretch="always")
        except Exception:
            pass

        right_paned = tk.PanedWindow(
            right,
            orient=tk.VERTICAL,
            sashrelief=tk.RAISED,
            sashwidth=6,
            sashpad=2,
            showhandle=True,
            handlesize=8,
            sashcursor="sb_v_double_arrow",
        )
        right_paned.grid(row=0, column=0, sticky="nsew")

        preview = ttk.Frame(right_paned)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        right_paned.add(preview)

        self.canvas = tk.Canvas(preview, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scroll = ttk.Scrollbar(preview, orient="horizontal", command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(preview, orient="vertical", command=self.canvas.yview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)
        self.canvas.bind("<MouseWheel>", self._on_preview_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_preview_shift_mousewheel)

        self._help_text = (
            "Workflow:\n"
            "1) Select an item (or Add ROI/Anchor)\n"
            "2) Click \"Draw ROI box\" or \"Pick anchor point\"\n"
            "3) Draw on the screenshot (drag for ROI; click for anchor)\n"
            "4) Save\n\n"
            "Preview controls:\n"
            "- Mouse wheel: scroll vertically\n"
            "- Shift + wheel: scroll horizontally\n"
            "- Ctrl + wheel: zoom\n\n"
            "Notes:\n"
            "- Coordinates are screen pixels (monitor-merged coordinate space).\n"
            "- Keep Nanonis window layout stable.\n"
        )

        bottom = ttk.Frame(right_paned)
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(1, weight=1)
        right_paned.add(bottom)
        try:
            right_paned.paneconfigure(preview, minsize=300, stretch="always")
            right_paned.paneconfigure(bottom, minsize=220, stretch="always")
        except Exception:
            pass
        self._help_btn = ttk.Button(bottom, text="Help", command=self._show_help_tooltip)
        self._help_btn.grid(row=0, column=0, sticky="e")

        form_container = ttk.Frame(bottom)
        form_container.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        form_container.columnconfigure(0, weight=1)
        form_container.rowconfigure(0, weight=1)

        self._form_canvas = tk.Canvas(form_container, highlightthickness=0)
        form_scroll = ttk.Scrollbar(form_container, orient="vertical", command=self._form_canvas.yview)
        self._form_canvas.configure(yscrollcommand=form_scroll.set)
        self._form_canvas.grid(row=0, column=0, sticky="nsew")
        form_scroll.grid(row=0, column=1, sticky="ns")

        self._form_inner = ttk.Frame(self._form_canvas)
        self._form_window = self._form_canvas.create_window((0, 0), window=self._form_inner, anchor="nw")
        self._form_inner.columnconfigure(0, weight=1)

        self._form_inner.bind("<Configure>", self._on_form_inner_configure)
        self._form_canvas.bind("<Configure>", self._on_form_canvas_configure)
        self._form_canvas.bind("<MouseWheel>", self._on_form_mousewheel)

        self.form = ttk.LabelFrame(self._form_inner, text="Selected item", padding=8)
        self.form.grid(row=0, column=0, sticky="nsew")
        self.form.columnconfigure(0, weight=1)
        self.form.columnconfigure(1, weight=3)
        self.form.rowconfigure(1, weight=1)

        self.kind_var = tk.StringVar(value="")
        ttk.Label(self.form, textvariable=self.kind_var).grid(row=0, column=0, columnspan=2, sticky="w")

        left_col = ttk.Frame(self.form)
        left_col.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left_col.columnconfigure(1, weight=1)

        right_col = ttk.Frame(self.form)
        right_col.grid(row=1, column=1, sticky="nsew")
        right_col.columnconfigure(0, weight=1)
        right_col.rowconfigure(1, weight=1)

        self._fields = {}
        field_order = ["name", "x", "y", "w", "h", "tags"]
        for i, field in enumerate(field_order):
            ttk.Label(left_col, text=field).grid(row=i, column=0, sticky="w", pady=2)
            ent = ttk.Entry(left_col, width=18)
            ent.configure(exportselection=False)
            ent.grid(row=i, column=1, sticky="ew", pady=2)
            ent.bind("<KeyRelease>", lambda _e, f=field: self._on_form_changed(source=f))
            ent.bind("<FocusOut>", lambda _e, f=field: self._on_form_changed(source=f))
            self._fields[field] = ent

        ttk.Label(right_col, text="description").grid(row=0, column=0, sticky="w")
        self._desc_text = tk.Text(right_col, height=8, wrap="word", exportselection=False)
        self._desc_text.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        self._desc_text.bind("<KeyRelease>", lambda _e: self._on_form_changed(source="description"))
        self._desc_text.bind("<FocusOut>", lambda _e: self._on_form_changed(source="description"))

        # Anchor-only: link ROIs that should be checked after using this anchor.
        self._linked_frame = ttk.LabelFrame(left_col, text="Linked ROIs (anchor only)", padding=8)
        linked_row = len(field_order)
        self._linked_frame.grid(row=linked_row, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self._linked_frame.columnconfigure(0, weight=1)
        self._linked_frame.rowconfigure(1, weight=1)
        left_col.rowconfigure(linked_row, weight=1)

        self._linked_roi_var = tk.StringVar(value="")
        self._linked_roi_combo = ttk.Combobox(self._linked_frame, textvariable=self._linked_roi_var, state="readonly")
        self._linked_roi_combo.grid(row=0, column=0, sticky="ew")
        self._linked_add_btn = ttk.Button(self._linked_frame, text="Add", command=self._add_linked_roi)
        self._linked_add_btn.grid(row=0, column=1, padx=(6, 0))

        self._linked_list = tk.Listbox(
            self._linked_frame, height=5, selectmode=tk.EXTENDED, exportselection=False
        )
        self._linked_list.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        self._linked_remove_btn = ttk.Button(self._linked_frame, text="Remove selected", command=self._remove_linked_roi)
        self._linked_remove_btn.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        self._set_linked_controls_enabled(False)

        self._refresh_roi_options()

    def _refresh_list(self) -> None:
        forced_rois = self._apply_forced_roi_activation()
        self.items_list.delete(0, tk.END)
        for r in self.rois:
            list_idx = self.items_list.size()
            self.items_list.insert(tk.END, self._format_item_label(kind="roi", name=r.name, active=r.active))
            if r.name in forced_rois:
                self.items_list.itemconfig(list_idx, foreground="#808080")
            else:
                self.items_list.itemconfig(list_idx, foreground=self._items_default_fg)
        for a in self.anchors:
            list_idx = self.items_list.size()
            self.items_list.insert(tk.END, self._format_item_label(kind="anchor", name=a.name, active=a.active))
            self.items_list.itemconfig(list_idx, foreground=self._items_default_fg)
        self._refresh_roi_options()

    def _format_item_label(self, *, kind: ItemKind, name: str, active: bool) -> str:
        box = "[x]" if active else "[ ]"
        tag = "[ROI]" if kind == "roi" else "[ANCHOR]"
        return f"{box} {tag} {name}"

    def _forced_roi_names(self) -> set[str]:
        roi_names = {r.name for r in self.rois if r.name}
        forced: set[str] = set()
        for a in self.anchors:
            if not a.active:
                continue
            for name in a.linked_rois or []:
                if name in roi_names:
                    forced.add(name)
        return forced

    def _apply_forced_roi_activation(self) -> set[str]:
        forced = self._forced_roi_names()
        if forced:
            for r in self.rois:
                if r.name in forced and not r.active:
                    r.active = True
        return forced

    def _list_index_to_item(self, list_idx: int) -> tuple[ItemKind, int]:
        if list_idx < len(self.rois):
            return ("roi", list_idx)
        return ("anchor", list_idx - len(self.rois))

    def _refresh_roi_options(self) -> None:
        try:
            values = [r.name for r in self.rois if r.name]
            self._linked_roi_combo["values"] = values
            if values and self._linked_roi_var.get() not in values:
                self._linked_roi_var.set(values[0])
            if not values:
                self._linked_roi_var.set("")
        except Exception:
            pass

    def _add_linked_roi(self) -> None:
        name = (self._linked_roi_var.get() or "").strip()
        if not name:
            return
        existing = set(self._linked_list.get(0, tk.END))
        if name in existing:
            return
        self._linked_list.insert(tk.END, name)
        self._on_form_changed(source="linked_rois")

    def _remove_linked_roi(self) -> None:
        sel = list(self._linked_list.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            self._linked_list.delete(idx)
        self._on_form_changed(source="linked_rois")

    def _set_linked_rois_ui(self, values: list[str]) -> None:
        self._linked_list.delete(0, tk.END)
        for v in values:
            if v:
                self._linked_list.insert(tk.END, v)

    def _set_linked_controls_enabled(self, enabled: bool) -> None:
        try:
            self._linked_frame.state(["!disabled"] if enabled else ["disabled"])
        except Exception:
            pass
        try:
            self._linked_roi_combo.configure(state="readonly" if enabled else "disabled")
        except Exception:
            pass
        try:
            self._linked_add_btn.state(["!disabled"] if enabled else ["disabled"])
        except Exception:
            pass
        try:
            self._linked_remove_btn.state(["!disabled"] if enabled else ["disabled"])
        except Exception:
            pass
        try:
            self._linked_list.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        except Exception:
            pass

    def _on_form_inner_configure(self, _e: tk.Event) -> None:
        if self._form_canvas is None:
            return
        try:
            self._form_canvas.configure(scrollregion=self._form_canvas.bbox("all"))
        except Exception:
            pass

    def _on_form_canvas_configure(self, _e: tk.Event) -> None:
        if self._form_canvas is None or self._form_window is None:
            return
        try:
            self._form_canvas.itemconfigure(self._form_window, width=self._form_canvas.winfo_width())
        except Exception:
            pass

    def _on_form_mousewheel(self, e: tk.Event) -> str:
        if self._form_canvas is None:
            return "break"
        delta = getattr(e, "delta", 0)
        if not delta:
            return "break"
        self._form_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
        return "break"

    def _on_preview_mousewheel(self, e: tk.Event) -> str:
        if self._screen_img is None:
            return "break"
        state = getattr(e, "state", 0)
        if state & 0x0004 or state & 0x0001:
            return "break"
        delta = getattr(e, "delta", 0)
        if not delta:
            return "break"
        self.canvas.yview_scroll(int(-1 * (delta / 120)), "units")
        return "break"

    def _on_preview_shift_mousewheel(self, e: tk.Event) -> str:
        if self._screen_img is None:
            return "break"
        state = getattr(e, "state", 0)
        if state & 0x0004:
            return "break"
        delta = getattr(e, "delta", 0)
        if not delta:
            return "break"
        self.canvas.xview_scroll(int(-1 * (delta / 120)), "units")
        return "break"

    def _show_help_tooltip(self) -> None:
        if self._help_window is not None and self._help_window.winfo_exists():
            self._hide_help_tooltip()
            return

        win = tk.Toplevel(self)
        win.title("Help")
        win.resizable(False, False)
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass

        ttk.Label(win, text=self._help_text, justify="left", padding=10).grid(row=0, column=0, sticky="w")
        win.update_idletasks()

        x = self.winfo_rootx() + 50
        y = self.winfo_rooty() + 50
        if self._help_btn is not None and self._help_btn.winfo_exists():
            x = self._help_btn.winfo_rootx()
            y = self._help_btn.winfo_rooty() + self._help_btn.winfo_height()
        win.geometry(f"+{x}+{y}")

        win.bind("<Escape>", lambda _e: self._hide_help_tooltip())
        win.bind("<FocusOut>", lambda _e: self._hide_help_tooltip())
        try:
            win.focus_set()
        except Exception:
            pass
        self._help_window = win

    def _hide_help_tooltip(self) -> None:
        if self._help_window is None:
            return
        try:
            self._help_window.destroy()
        except Exception:
            pass
        self._help_window = None

    def _selected(self) -> tuple[Optional[ItemKind], Optional[int]]:
        sel = self.items_list.curselection()
        if not sel:
            return (None, None)
        idx = int(sel[0])
        if idx < len(self.rois):
            return ("roi", idx)
        return ("anchor", idx - len(self.rois))

    def _on_items_click(self, e: tk.Event) -> Optional[str]:
        idx = self.items_list.nearest(e.y)
        if idx < 0:
            return None
        bbox = self.items_list.bbox(idx)
        if not bbox:
            return None
        x0, _y0, _w, _h = bbox
        if e.x > x0 + 22:
            return None

        kind, item_idx = self._list_index_to_item(idx)
        forced = self._forced_roi_names()
        if kind == "roi":
            item = self.rois[item_idx]
            if item.name in forced:
                self.items_list.selection_clear(0, tk.END)
                self.items_list.selection_set(idx)
                self._on_select(populate_form=True)
                return "break"
            item.active = not item.active
        else:
            item = self.anchors[item_idx]
            item.active = not item.active

        self._refresh_list()
        self.items_list.selection_clear(0, tk.END)
        self.items_list.selection_set(idx)
        self._on_select(populate_form=True)
        return "break"

    def _on_select(self, *, populate_form: bool = True) -> None:
        kind, idx = self._selected()
        self._clear_overlay()
        if kind is None or idx is None:
            if populate_form:
                self.kind_var.set("")
                for e in self._fields.values():
                    e.delete(0, tk.END)
                self._set_description("")
            return

        if populate_form:
            self._suppress_form_events = True
            if kind == "roi":
                item = self.rois[idx]
                self.kind_var.set("ROI (Observation)")
                self._set_field("name", item.name)
                self._set_field("x", str(item.x))
                self._set_field("y", str(item.y))
                self._set_field("w", str(item.w))
                self._set_field("h", str(item.h))
                self._set_field("description", item.description)
                self._set_field("tags", item.tags)
                self._set_linked_controls_enabled(False)
                self._set_linked_rois_ui([])
            else:
                item = self.anchors[idx]
                self.kind_var.set("Anchor (Action click point)")
                self._set_field("name", item.name)
                self._set_field("x", str(item.x))
                self._set_field("y", str(item.y))
                self._set_field("w", "")
                self._set_field("h", "")
                self._set_field("description", item.description)
                self._set_field("tags", item.tags)
                self._set_linked_controls_enabled(True)
                self._set_linked_rois_ui(list(item.linked_rois or []))
            self._suppress_form_events = False

        if kind == "roi":
            self._draw_existing_roi(self.rois[idx])
        else:
            self._draw_existing_anchor(self.anchors[idx])

    def _set_field(self, key: str, value: str) -> None:
        if key == "description":
            self._set_description(value)
            return
        ent = self._fields[key]
        ent.delete(0, tk.END)
        ent.insert(0, value)

    def _set_description(self, value: str) -> None:
        if self._desc_text is None:
            return
        self._desc_text.delete("1.0", tk.END)
        self._desc_text.insert("1.0", value or "")

    def _get_description(self) -> str:
        if self._desc_text is None:
            return ""
        return self._desc_text.get("1.0", tk.END).rstrip("\n")

    def _try_int(self, value: str) -> Optional[int]:
        s = (value or "").strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None

    def _update_selected_list_label(self, *, kind: ItemKind, idx: int) -> None:
        list_idx = idx if kind == "roi" else len(self.rois) + idx
        if kind == "roi":
            item = self.rois[idx]
            label = self._format_item_label(kind="roi", name=item.name, active=item.active)
        else:
            item = self.anchors[idx]
            label = self._format_item_label(kind="anchor", name=item.name, active=item.active)
        try:
            self.items_list.delete(list_idx)
            self.items_list.insert(list_idx, label)
            forced = self._forced_roi_names()
            if kind == "roi" and item.name in forced:
                self.items_list.itemconfig(list_idx, foreground="#808080")
            else:
                self.items_list.itemconfig(list_idx, foreground=self._items_default_fg)
            self.items_list.selection_clear(0, tk.END)
            self.items_list.selection_set(list_idx)
        except Exception:
            pass

    def _on_form_changed(self, *, source: str) -> None:
        if self._suppress_form_events:
            return
        kind, idx = self._selected()
        if kind is None or idx is None:
            return

        name_in = self._fields["name"].get().strip()
        x_in = self._try_int(self._fields["x"].get())
        y_in = self._try_int(self._fields["y"].get())
        description_in = self._get_description()
        tags_in = self._fields["tags"].get()

        if kind == "roi":
            cur = self.rois[idx]
            w_in = self._try_int(self._fields["w"].get())
            h_in = self._try_int(self._fields["h"].get())
            self.rois[idx] = RoiDraft(
                name=name_in or cur.name,
                x=cur.x if x_in is None else x_in,
                y=cur.y if y_in is None else y_in,
                w=cur.w if (w_in is None or w_in <= 0) else w_in,
                h=cur.h if (h_in is None or h_in <= 0) else h_in,
                description=description_in,
                tags=tags_in,
                active=cur.active,
            )
            if source in {"name"}:
                self._refresh_roi_options()
                self._update_selected_list_label(kind="roi", idx=idx)
            if source in {"x", "y", "w", "h"}:
                self._clear_overlay()
                self._draw_existing_roi(self.rois[idx])
        else:
            cur = self.anchors[idx]
            linked = [str(v) for v in self._linked_list.get(0, tk.END) if str(v).strip()]
            self.anchors[idx] = AnchorDraft(
                name=name_in or cur.name,
                x=cur.x if x_in is None else x_in,
                y=cur.y if y_in is None else y_in,
                description=description_in,
                tags=tags_in,
                linked_rois=linked,
                active=cur.active,
            )
            if source in {"name"}:
                self._update_selected_list_label(kind="anchor", idx=idx)
            if source in {"x", "y"}:
                self._clear_overlay()
                self._draw_existing_anchor(self.anchors[idx])
            if source in {"linked_rois"}:
                self._refresh_list()
                self.items_list.selection_clear(0, tk.END)
                self.items_list.selection_set(len(self.rois) + idx)

    def _add_roi(self) -> None:
        existing = {r.name for r in self.rois}
        name = _dedupe_name(existing, "new_roi")
        self.rois.append(RoiDraft(name=name, description=""))
        self._refresh_list()
        self.items_list.selection_clear(0, tk.END)
        self.items_list.selection_set(len(self.rois) - 1)
        self._on_select()

    def _add_anchor(self) -> None:
        existing = {a.name for a in self.anchors}
        name = _dedupe_name(existing, "new_anchor")
        self.anchors.append(AnchorDraft(name=name, description=""))
        self._refresh_list()
        self.items_list.selection_clear(0, tk.END)
        self.items_list.selection_set(len(self.rois) + len(self.anchors) - 1)
        self._on_select()

    def _delete_selected(self) -> None:
        kind, idx = self._selected()
        if kind is None or idx is None:
            return
        if not messagebox.askyesno("Delete", "Delete selected item?"):
            return
        if kind == "roi":
            removed = self.rois.pop(idx)
            removed_name = (removed.name or "").strip()
            if removed_name:
                for a in self.anchors:
                    a.linked_rois = [x for x in (a.linked_rois or []) if x != removed_name]
        else:
            self.anchors.pop(idx)
        self._refresh_list()
        self._on_select()

    def _save(self) -> None:
        try:
            self._apply_forced_roi_activation()
            names: set[str] = set()
            roi_names: set[str] = set()
            for r in self.rois:
                if not r.name:
                    raise ValueError("ROI name cannot be empty")
                if r.name in names:
                    raise ValueError(f"Duplicate name: {r.name!r}")
                names.add(r.name)
                roi_names.add(r.name)
                if r.w <= 0 or r.h <= 0:
                    raise ValueError(f"ROI {r.name!r} must have positive w/h")
            for a in self.anchors:
                if not a.name:
                    raise ValueError("Anchor name cannot be empty")
                if a.name in names:
                    raise ValueError(f"Duplicate name: {a.name!r}")
                names.add(a.name)
                a.linked_rois = [x for x in (a.linked_rois or []) if x in roi_names]

            out = dict(self.raw)
            out["rois"] = [r.to_json() for r in self.rois]
            out["anchors"] = [a.to_json() for a in self.anchors]
            self.workspace_path.write_text(json.dumps(out, indent=2, sort_keys=False), encoding="utf-8")
            messagebox.showinfo("Saved", f"Saved to {self.workspace_path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _begin_draw_roi(self) -> None:
        kind, idx = self._selected()
        if kind != "roi" or idx is None:
            messagebox.showinfo("Select ROI", "Select an ROI item first (or Add ROI).")
            return
        self.mode = "draw_roi"
        self._drawing_item_index = idx
        self._clear_overlay()
        self._status("Draw ROI: click+drag on screenshot")

    def _begin_pick_anchor(self) -> None:
        kind, idx = self._selected()
        if kind != "anchor" or idx is None:
            messagebox.showinfo("Select anchor", "Select an Anchor item first (or Add Anchor).")
            return
        self.mode = "pick_anchor"
        self._drawing_item_index = idx
        self._clear_overlay()
        self._status("Pick anchor: click on screenshot")

    def _status(self, text: str) -> None:
        self.title(f"Workspace Calibrator - {text}")

    def _refresh_screenshot(self) -> None:
        self._clear_overlay()
        capturer = ScreenCapturer(monitor_index=0)
        img = capturer.capture_fullscreen()

        # mss monitor 0 is the bounding box of all monitors (can have negative left/top).
        try:
            import mss

            with mss.mss() as sct:
                mon0 = sct.monitors[0]
                left = int(mon0.get("left", 0))
                top = int(mon0.get("top", 0))
                width = int(mon0.get("width", img.width))
                height = int(mon0.get("height", img.height))
        except Exception:
            left, top, width, height = 0, 0, img.width, img.height

        self._screen_img = img
        self._screen_bbox = (left, top, width, height)
        self._render_canvas_image()
        self._on_select(populate_form=False)

    def _render_canvas_image(self) -> None:
        if self._screen_img is None:
            return

        self.update_idletasks()
        canvas_w = max(200, int(self.canvas.winfo_width()))
        canvas_h = max(200, int(self.canvas.winfo_height()))
        img_w, img_h = self._screen_img.size

        sx = canvas_w / img_w
        sy = canvas_h / img_h
        self._fit_scale = min(sx, sy, 1.0)
        self._scale = max(0.05, self._fit_scale * self._zoom)

        if self._scale < 1.0:
            resized = self._screen_img.resize(
                (int(img_w * self._scale), int(img_h * self._scale)), resample=Image.Resampling.LANCZOS
            )
        else:
            resized = self._screen_img

        self._screen_photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._screen_photo, anchor="nw", tags="bg")
        self.canvas.config(scrollregion=(0, 0, resized.width, resized.height))
        self._on_select(populate_form=False)

    def _on_canvas_resize(self, _e: tk.Event) -> None:
        if self._render_job is not None:
            try:
                self.after_cancel(self._render_job)
            except Exception:
                pass
        self._render_job = self.after(75, self._render_canvas_image)

    def _canvas_to_screen(self, cx: int, cy: int) -> tuple[int, int]:
        left, top, _, _ = self._screen_bbox
        if self._scale <= 0:
            return left, top
        x = int(left + (cx / self._scale))
        y = int(top + (cy / self._scale))
        return x, y

    def _screen_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        left, top, _, _ = self._screen_bbox
        if self._scale <= 0:
            return 0, 0
        cx = int((x - left) * self._scale)
        cy = int((y - top) * self._scale)
        return cx, cy

    def _clear_overlay(self) -> None:
        if self._active_rect_id is not None:
            try:
                self.canvas.delete(self._active_rect_id)
            except Exception:
                pass
        for item in self._active_crosshair:
            try:
                self.canvas.delete(item)
            except Exception:
                pass
        self._active_rect_id = None
        self._active_crosshair = []

    def _draw_existing_roi(self, roi: RoiDraft) -> None:
        if self._screen_img is None:
            return
        cx1, cy1 = self._screen_to_canvas(roi.x, roi.y)
        cx2, cy2 = self._screen_to_canvas(roi.x + roi.w, roi.y + roi.h)
        self._active_rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00d1ff", width=2)

    def _draw_existing_anchor(self, anchor: AnchorDraft) -> None:
        if self._screen_img is None:
            return
        cx, cy = self._screen_to_canvas(anchor.x, anchor.y)
        r = 6
        self._active_crosshair = [
            self.canvas.create_line(cx - r, cy, cx + r, cy, fill="#ffcc00", width=2),
            self.canvas.create_line(cx, cy - r, cx, cy + r, fill="#ffcc00", width=2),
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#ffcc00", width=2),
        ]

    def _on_zoom(self, e: tk.Event) -> str:
        if self.mode in {"draw_roi", "pick_anchor"}:
            return "break"
        if self._screen_img is None:
            return "break"
        if getattr(e, "delta", 0) == 0:
            return "break"

        old_scale = self._scale if self._scale > 0 else 1.0
        direction = 1 if e.delta > 0 else -1
        zoom_factor = 1.1 if direction > 0 else (1 / 1.1)
        new_zoom = self._zoom * zoom_factor
        new_zoom = max(0.25, min(6.0, new_zoom))
        if abs(new_zoom - self._zoom) < 1e-6:
            return "break"

        canvas = self.canvas
        pointer_x = canvas.canvasx(int(getattr(e, "x", 0)))
        pointer_y = canvas.canvasy(int(getattr(e, "y", 0)))
        view_x = canvas.canvasx(0)
        view_y = canvas.canvasy(0)
        offset_x = pointer_x - view_x
        offset_y = pointer_y - view_y

        self._zoom = new_zoom
        self._render_canvas_image()

        new_scale = self._scale if self._scale > 0 else 1.0
        scale_ratio = new_scale / old_scale
        new_pointer_x = pointer_x * scale_ratio
        new_pointer_y = pointer_y * scale_ratio
        new_view_x = new_pointer_x - offset_x
        new_view_y = new_pointer_y - offset_y
        self._scroll_canvas_to(new_view_x, new_view_y)
        return "break"

    def _scroll_canvas_to(self, x: float, y: float) -> None:
        try:
            width = float(self.canvas.winfo_width())
            height = float(self.canvas.winfo_height())
            region = self.canvas.bbox("all")
            if not region:
                return
            x0, y0, x1, y1 = region
            region_w = float(x1 - x0)
            region_h = float(y1 - y0)
            if region_w > width:
                x = max(0.0, min(x, region_w - width))
                self.canvas.xview_moveto(x / region_w)
            else:
                self.canvas.xview_moveto(0.0)
            if region_h > height:
                y = max(0.0, min(y, region_h - height))
                self.canvas.yview_moveto(y / region_h)
            else:
                self.canvas.yview_moveto(0.0)
        except Exception:
            pass

    def _on_mouse_down(self, e: tk.Event) -> None:
        if self._screen_img is None:
            return
        cx = int(self.canvas.canvasx(e.x))
        cy = int(self.canvas.canvasy(e.y))
        if self.mode == "draw_roi":
            self._draw_start = (cx, cy)
            self._clear_overlay()
            self._active_rect_id = self.canvas.create_rectangle(cx, cy, cx, cy, outline="#00d1ff", width=2)
        elif self.mode == "pick_anchor":
            if self._drawing_item_index is None:
                return
            sx, sy = self._canvas_to_screen(cx, cy)
            idx = self._drawing_item_index
            a = self.anchors[idx]
            self.anchors[idx] = AnchorDraft(
                name=a.name,
                x=sx,
                y=sy,
                description=a.description,
                tags=a.tags,
                linked_rois=list(a.linked_rois or []),
                active=a.active,
            )
            self._refresh_list()
            self.items_list.selection_clear(0, tk.END)
            self.items_list.selection_set(len(self.rois) + idx)
            self.mode = "idle"
            self._status("idle")
            self._on_select(populate_form=True)

    def _on_mouse_move(self, e: tk.Event) -> None:
        if self.mode != "draw_roi" or self._draw_start is None or self._active_rect_id is None:
            return
        cx = int(self.canvas.canvasx(e.x))
        cy = int(self.canvas.canvasy(e.y))
        x0, y0 = self._draw_start
        self.canvas.coords(self._active_rect_id, x0, y0, cx, cy)

    def _on_mouse_up(self, e: tk.Event) -> None:
        if self.mode != "draw_roi" or self._draw_start is None:
            return
        if self._drawing_item_index is None:
            return

        x0, y0 = self._draw_start
        x1, y1 = int(self.canvas.canvasx(e.x)), int(self.canvas.canvasy(e.y))
        self._draw_start = None

        cx1, cy1 = min(x0, x1), min(y0, y1)
        cx2, cy2 = max(x0, x1), max(y0, y1)

        sx1, sy1 = self._canvas_to_screen(cx1, cy1)
        sx2, sy2 = self._canvas_to_screen(cx2, cy2)
        w = max(1, sx2 - sx1)
        h = max(1, sy2 - sy1)

        idx = self._drawing_item_index
        r = self.rois[idx]
        self.rois[idx] = RoiDraft(
            name=r.name,
            x=sx1,
            y=sy1,
            w=w,
            h=h,
            description=r.description,
            tags=r.tags,
            active=r.active,
        )

        self._refresh_list()
        self.items_list.selection_clear(0, tk.END)
        self.items_list.selection_set(idx)
        self.mode = "idle"
        self._status("idle")
        self._on_select(populate_form=True)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="GUI tool to define ROIs/anchors and save workspace.json.")
    ap.add_argument("--workspace", default="workspace.json", help="Path to workspace.json")
    args = ap.parse_args(argv)

    try:
        _enable_windows_dpi_awareness()
        app = CalibratorApp(workspace_path=Path(args.workspace))
        app.mainloop()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
