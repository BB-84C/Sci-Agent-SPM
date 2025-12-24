from __future__ import annotations

import argparse
import ctypes
import json
import sys
import tkinter as tk
from dataclasses import dataclass
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

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "description": self.description,
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

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "x": int(self.x),
            "y": int(self.y),
            "description": self.description,
        }
        tags = [t.strip() for t in (self.tags or "").split(",") if t.strip()]
        if tags:
            out["tags"] = tags
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
                    )
                )

        self.anchors: list[AnchorDraft] = []
        for a in self.raw.get("anchors", []):
            if isinstance(a, dict):
                self.anchors.append(
                    AnchorDraft(
                        name=str(a.get("name", "")),
                        x=int(a.get("x", 0)),
                        y=int(a.get("y", 0)),
                        description=str(a.get("description", "")),
                        tags=",".join(str(t) for t in (a.get("tags") or []) if isinstance(t, (str, int, float))),
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
        self._render_job: Optional[str] = None

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
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsw")

        ttk.Label(left, text="Items").grid(row=0, column=0, sticky="w")
        self.items_list = tk.Listbox(left, width=36, height=28)
        self.items_list.grid(row=1, column=0, sticky="nsew")
        self.items_list.bind("<<ListboxSelect>>", lambda _e: self._on_select(populate_form=True))

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

        self.form = ttk.LabelFrame(left, text="Selected item", padding=8)
        self.form.grid(row=4, column=0, sticky="ew", pady=(10, 0))

        self.kind_var = tk.StringVar(value="")
        ttk.Label(self.form, textvariable=self.kind_var).grid(row=0, column=0, columnspan=2, sticky="w")

        self._fields: dict[str, tk.Entry] = {}
        for i, field in enumerate(["name", "x", "y", "w", "h", "description", "tags"], start=1):
            ttk.Label(self.form, text=field).grid(row=i, column=0, sticky="w", pady=2)
            ent = ttk.Entry(self.form, width=30)
            ent.configure(exportselection=False)
            ent.grid(row=i, column=1, sticky="ew", pady=2)
            self._fields[field] = ent

        self.form.columnconfigure(1, weight=1)

        apply_row = 1 + 7
        ttk.Button(self.form, text="Apply changes", command=self._apply_form).grid(
            row=apply_row, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(right, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        help_txt = (
            "Workflow:\n"
            "1) Select an item (or Add ROI/Anchor)\n"
            "2) Click “Draw ROI box” or “Pick anchor point”\n"
            "3) Draw on the screenshot (drag for ROI; click for anchor)\n"
            "4) Save\n\n"
            "Notes:\n"
            "- Coordinates are screen pixels (monitor-merged coordinate space).\n"
            "- Keep Nanonis window layout stable.\n"
        )
        ttk.Label(right, text=help_txt, justify="left").grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _refresh_list(self) -> None:
        self.items_list.delete(0, tk.END)
        for r in self.rois:
            self.items_list.insert(tk.END, f"[ROI] {r.name}")
        for a in self.anchors:
            self.items_list.insert(tk.END, f"[ANCHOR] {a.name}")

    def _selected(self) -> tuple[Optional[ItemKind], Optional[int]]:
        sel = self.items_list.curselection()
        if not sel:
            return (None, None)
        idx = int(sel[0])
        if idx < len(self.rois):
            return ("roi", idx)
        return ("anchor", idx - len(self.rois))

    def _on_select(self, *, populate_form: bool = True) -> None:
        kind, idx = self._selected()
        self._clear_overlay()
        if kind is None or idx is None:
            if populate_form:
                self.kind_var.set("")
                for e in self._fields.values():
                    e.delete(0, tk.END)
            return

        if populate_form:
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

        if kind == "roi":
            self._draw_existing_roi(self.rois[idx])
        else:
            self._draw_existing_anchor(self.anchors[idx])

    def _set_field(self, key: str, value: str) -> None:
        ent = self._fields[key]
        ent.delete(0, tk.END)
        ent.insert(0, value)

    def _apply_form(self) -> None:
        kind, idx = self._selected()
        if kind is None or idx is None:
            messagebox.showinfo("No selection", "Select an ROI or Anchor first.")
            return
        try:
            name = self._fields["name"].get().strip()
            if not name:
                raise ValueError("name is required")
            x = _safe_int(self._fields["x"].get(), field="x")
            y = _safe_int(self._fields["y"].get(), field="y")
            description = self._fields["description"].get().strip()
            tags = self._fields["tags"].get().strip()
            if kind == "roi":
                w = _safe_int(self._fields["w"].get(), field="w")
                h = _safe_int(self._fields["h"].get(), field="h")
                if w <= 0 or h <= 0:
                    raise ValueError("ROI must have positive w/h")
                self.rois[idx] = RoiDraft(name=name, x=x, y=y, w=w, h=h, description=description, tags=tags)
            else:
                self.anchors[idx] = AnchorDraft(name=name, x=x, y=y, description=description, tags=tags)

            self._refresh_list()
            self.items_list.selection_clear(0, tk.END)
            self.items_list.selection_set(idx if kind == "roi" else len(self.rois) + idx)
            self._on_select()
        except Exception as e:
            messagebox.showerror("Invalid values", str(e))

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
            self.rois.pop(idx)
        else:
            self.anchors.pop(idx)
        self._refresh_list()
        self._on_select()

    def _save(self) -> None:
        try:
            names: set[str] = set()
            for r in self.rois:
                if not r.name:
                    raise ValueError("ROI name cannot be empty")
                if r.name in names:
                    raise ValueError(f"Duplicate name: {r.name!r}")
                names.add(r.name)
                if r.w <= 0 or r.h <= 0:
                    raise ValueError(f"ROI {r.name!r} must have positive w/h")
            for a in self.anchors:
                if not a.name:
                    raise ValueError("Anchor name cannot be empty")
                if a.name in names:
                    raise ValueError(f"Duplicate name: {a.name!r}")
                names.add(a.name)

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
        self._scale = min(sx, sy, 1.0)

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
        x = int(left + (cx / self._scale))
        y = int(top + (cy / self._scale))
        return x, y

    def _screen_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        left, top, _, _ = self._screen_bbox
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

    def _on_mouse_down(self, e: tk.Event) -> None:
        if self._screen_img is None:
            return
        if self.mode == "draw_roi":
            self._draw_start = (int(e.x), int(e.y))
            self._clear_overlay()
            self._active_rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#00d1ff", width=2)
        elif self.mode == "pick_anchor":
            if self._drawing_item_index is None:
                return
            sx, sy = self._canvas_to_screen(int(e.x), int(e.y))
            idx = self._drawing_item_index
            a = self.anchors[idx]
            self.anchors[idx] = AnchorDraft(name=a.name, x=sx, y=sy, description=a.description, tags=a.tags)
            self._refresh_list()
            self.items_list.selection_clear(0, tk.END)
            self.items_list.selection_set(len(self.rois) + idx)
            self.mode = "idle"
            self._status("idle")
            self._on_select(populate_form=False)

    def _on_mouse_move(self, e: tk.Event) -> None:
        if self.mode != "draw_roi" or self._draw_start is None or self._active_rect_id is None:
            return
        x0, y0 = self._draw_start
        self.canvas.coords(self._active_rect_id, x0, y0, int(e.x), int(e.y))

    def _on_mouse_up(self, e: tk.Event) -> None:
        if self.mode != "draw_roi" or self._draw_start is None:
            return
        if self._drawing_item_index is None:
            return

        x0, y0 = self._draw_start
        x1, y1 = int(e.x), int(e.y)
        self._draw_start = None

        cx1, cy1 = min(x0, x1), min(y0, y1)
        cx2, cy2 = max(x0, x1), max(y0, y1)

        sx1, sy1 = self._canvas_to_screen(cx1, cy1)
        sx2, sy2 = self._canvas_to_screen(cx2, cy2)
        w = max(1, sx2 - sx1)
        h = max(1, sy2 - sy1)

        idx = self._drawing_item_index
        r = self.rois[idx]
        self.rois[idx] = RoiDraft(name=r.name, x=sx1, y=sy1, w=w, h=h, description=r.description, tags=r.tags)

        self._refresh_list()
        self.items_list.selection_clear(0, tk.END)
        self.items_list.selection_set(idx)
        self.mode = "idle"
        self._status("idle")
        self._on_select(populate_form=False)


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
