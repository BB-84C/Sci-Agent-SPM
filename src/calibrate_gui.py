from __future__ import annotations

import argparse
import ctypes
import json
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Literal, Optional

from PIL import Image, ImageTk

from .capture import ScreenCapturer
from .tui_settings import DEFAULT_SETTINGS_PATH, load_tui_settings, merge_settings, save_tui_settings


ItemKind = Literal["roi", "anchor", "group"]


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
    group: str = ""

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
        if self.group:
            out["group"] = self.group
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
    group: str = ""

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
        if self.group:
            out["group"] = self.group
        return out


@dataclass
class GroupDraft:
    name: str = ""
    description: str = ""
    tags: str = ""
    active: bool = True
    group: str = ""
    collapsed: bool = False

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "active": bool(self.active),
        }
        tags = [t.strip() for t in (self.tags or "").split(",") if t.strip()]
        if tags:
            out["tags"] = tags
        if self.group:
            out["group"] = self.group
        return out


def _load_workspace_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rois": [], "anchors": [], "tools": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("workspace must be a JSON object")
    raw.setdefault("rois", [])
    raw.setdefault("anchors", [])
    raw.setdefault("groups", [])
    raw.setdefault("tools", {})
    if (
        not isinstance(raw["rois"], list)
        or not isinstance(raw["anchors"], list)
        or not isinstance(raw["groups"], list)
        or not isinstance(raw["tools"], dict)
    ):
        raise ValueError("workspace fields must be {rois:list, anchors:list, groups:list, tools:object}")
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

        self.repo_root = Path(__file__).resolve().parent.parent
        self._settings_path = self.repo_root / DEFAULT_SETTINGS_PATH

        self.workspace_path = workspace_path
        self.workspaces_dir = self.workspace_path.parent / "workspaces"
        try:
            self.workspaces_dir.mkdir(exist_ok=True)
        except Exception:
            pass
        self.raw = _load_workspace_raw(workspace_path)
        self._load_workspace_data(self.raw)

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
        self._list_index_map: list[tuple[ItemKind, int]] = []
        self._group_var: Optional[tk.StringVar] = None
        self._filter_tag_vars: dict[str, tk.BooleanVar] = {}
        self._filter_keyword_var: Optional[tk.StringVar] = None
        self._filter_logic_var: Optional[tk.StringVar] = None
        self._tag_frame_inner: Optional[ttk.Frame] = None
        self._tag_canvas: Optional[tk.Canvas] = None
        self._tag_window: Optional[int] = None
        self._group_combo: Optional[ttk.Combobox] = None
        self._group_display_to_name: dict[str, str] = {}
        self._group_name_to_display: dict[str, str] = {}
        self._entry_history: dict[tk.Entry, list[tuple[str, int]]] = {}
        self._entry_history_index: dict[tk.Entry, int] = {}
        self._entry_field_map: dict[tk.Entry, str] = {}
        self._entry_history_lock: bool = False

        self._build_ui()
        self._refresh_screenshot()
        self._refresh_list()
        self._status("idle")

    def _load_workspace_data(self, raw: dict[str, Any]) -> None:
        self.rois = []
        for r in raw.get("rois", []):
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
                        group=str(r.get("group", "")),
                    )
                )

        self.anchors = []
        for a in raw.get("anchors", []):
            if isinstance(a, dict):
                linked = a.get("linked_ROIs", []) or []
                linked_list = (
                    [str(x) for x in linked if isinstance(x, (str, int, float)) and str(x).strip()]
                    if isinstance(linked, list)
                    else []
                )
                self.anchors.append(
                    AnchorDraft(
                        name=str(a.get("name", "")),
                        x=int(a.get("x", 0)),
                        y=int(a.get("y", 0)),
                        description=str(a.get("description", "")),
                        tags=",".join(str(t) for t in (a.get("tags") or []) if isinstance(t, (str, int, float))),
                        linked_rois=linked_list,
                        active=_parse_active(a.get("active", True)),
                        group=str(a.get("group", "")),
                    )
                )

        self.groups = []
        for g in raw.get("groups", []):
            if isinstance(g, dict):
                self.groups.append(
                    GroupDraft(
                        name=str(g.get("name", "")),
                        description=str(g.get("description", "")),
                        tags=",".join(str(t) for t in (g.get("tags") or []) if isinstance(t, (str, int, float))),
                        active=_parse_active(g.get("active", True)),
                        group=str(g.get("group", "")),
                    )
                )

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

        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load workspace...", command=self._load_workspace_dialog)
        file_menu.add_command(label="Export workspace...", command=self._export_workspace_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Set agent workspace...", command=self._set_agent_workspace_dialog)
        file_menu.add_command(label="Use current workspace for agent", command=self._set_agent_workspace_current)
        menubar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Show help", command=self._show_help_tooltip)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=menubar)

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

        self.items_list = tk.Listbox(
            items_frame, width=36, height=14, exportselection=False, selectmode=tk.EXTENDED
        )
        items_scroll = ttk.Scrollbar(items_frame, orient="vertical", command=self.items_list.yview)
        self.items_list.configure(yscrollcommand=items_scroll.set)
        self.items_list.grid(row=0, column=0, sticky="nsew")
        items_scroll.grid(row=0, column=1, sticky="ns")
        self.items_list.bind("<<ListboxSelect>>", lambda _e: self._on_select(populate_form=True))
        self.items_list.bind("<Button-1>", self._on_items_click, add="+")
        self.items_list.bind("<Double-Button-1>", self._on_items_double_click)
        self._items_default_fg = str(self.items_list.cget("fg"))

        filter_frame = ttk.LabelFrame(left, text="Filter", padding=8)
        filter_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        filter_frame.columnconfigure(0, weight=1)
        filter_frame.rowconfigure(1, weight=1)

        ttk.Label(filter_frame, text="Tags").grid(row=0, column=0, sticky="w")
        self._tag_canvas = tk.Canvas(filter_frame, height=100, highlightthickness=0)
        tag_scroll = ttk.Scrollbar(filter_frame, orient="vertical", command=self._tag_canvas.yview)
        self._tag_canvas.configure(yscrollcommand=tag_scroll.set)
        self._tag_canvas.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        tag_scroll.grid(row=1, column=1, sticky="ns", pady=(2, 0))

        self._tag_frame_inner = ttk.Frame(self._tag_canvas)
        self._tag_window = self._tag_canvas.create_window((0, 0), window=self._tag_frame_inner, anchor="nw")
        self._tag_frame_inner.bind("<Configure>", self._on_tag_frame_configure)
        self._tag_canvas.bind("<Configure>", self._on_tag_canvas_configure)

        keyword_row = ttk.Frame(filter_frame)
        keyword_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        keyword_row.columnconfigure(1, weight=1)
        ttk.Label(keyword_row, text="Keyword").grid(row=0, column=0, sticky="w")
        self._filter_keyword_var = tk.StringVar(value="")
        keyword_entry = ttk.Entry(keyword_row, textvariable=self._filter_keyword_var)
        keyword_entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        self._filter_logic_var = tk.StringVar(value="AND")
        ttk.Button(keyword_row, textvariable=self._filter_logic_var, command=self._toggle_filter_logic, width=4).grid(
            row=0, column=2
        )
        ttk.Button(keyword_row, text="Clear", command=self._clear_filters).grid(row=0, column=3, padx=(6, 0))
        self._filter_keyword_var.trace_add("write", lambda *_: self._refresh_list())

        btns = ttk.Frame(left)
        btns.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        ttk.Button(btns, text="Add ROI", command=self._add_roi).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(btns, text="Add Anchor", command=self._add_anchor).grid(row=0, column=1, sticky="ew")
        ttk.Button(btns, text="Add Group", command=self._add_group).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(btns, text="Delete", command=self._delete_selected).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(btns, text="Save", command=self._save).grid(row=2, column=1, sticky="ew", pady=(6, 0))

        tools = ttk.LabelFrame(left, text="Pick on screenshot", padding=8)
        tools.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(tools, text="Draw ROI box", command=self._begin_draw_roi).grid(row=0, column=0, sticky="ew")
        ttk.Button(tools, text="Pick anchor point", command=self._begin_pick_anchor).grid(
            row=1, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Button(tools, text="Refresh screenshot", command=self._refresh_screenshot).grid(
            row=2, column=0, sticky="ew", pady=(6, 0)
        )

        preview_panel = ttk.Frame(paned, padding=8)
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(0, weight=1)

        selected_panel = ttk.Frame(paned, padding=8)
        selected_panel.columnconfigure(0, weight=1)
        selected_panel.rowconfigure(0, weight=1)

        paned.add(preview_panel)
        paned.add(left)
        paned.add(selected_panel)
        try:
            paned.paneconfigure(preview_panel, minsize=600, stretch="always")
            paned.paneconfigure(left, minsize=260, stretch="never")
            paned.paneconfigure(selected_panel, minsize=360, stretch="always")
        except Exception:
            pass

        preview = ttk.Frame(preview_panel)
        preview.grid(row=0, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)

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
        self.canvas.bind("<Shift-MouseWheel>", self._on_preview_shift_mousewheel)
        self.canvas.bind("<Alt-MouseWheel>", self._on_preview_alt_mousewheel)
        self.canvas.bind("<MouseWheel>", self._on_preview_mousewheel)

        self._help_text = (
            "Workflow:\n"
            "1) Select an item (or Add ROI/Anchor)\n"
            "2) Click \"Draw ROI box\" or \"Pick anchor point\"\n"
            "3) Draw on the screenshot (drag for ROI; click for anchor)\n"
            "4) Save\n\n"
            "Preview controls:\n"
            "- Mouse wheel: scroll vertically\n"
            "- Alt/Shift + wheel: scroll horizontally\n"
            "- Ctrl + wheel: zoom\n\n"
            "Notes:\n"
            "- Coordinates are screen pixels (monitor-merged coordinate space).\n"
            "- Keep the instrument control window layout stable.\n"
        )

        form_container = ttk.Frame(selected_panel)
        form_container.grid(row=0, column=0, sticky="nsew")
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
        self._form_inner.rowconfigure(0, weight=1)
        self._form_inner.grid_propagate(False)

        self._form_inner.bind("<Configure>", self._on_form_inner_configure)
        self._form_canvas.bind("<Configure>", self._on_form_canvas_configure)
        self._form_canvas.bind("<MouseWheel>", self._on_form_mousewheel)

        self.form = ttk.LabelFrame(self._form_inner, text="Selected item", padding=8)
        self.form.grid(row=0, column=0, sticky="nsew")
        self.form.columnconfigure(0, weight=1)
        self.form.rowconfigure(2, weight=1)

        self.kind_var = tk.StringVar(value="")
        ttk.Label(self.form, textvariable=self.kind_var).grid(row=0, column=0, sticky="w")

        top_row = ttk.Frame(self.form)
        top_row.grid(row=1, column=0, sticky="nsew")
        top_row.columnconfigure(1, weight=1)

        desc_row = ttk.Frame(self.form)
        desc_row.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        desc_row.columnconfigure(0, weight=1)
        desc_row.rowconfigure(1, weight=1)

        self._fields = {}
        field_order = ["name", "x", "y", "w", "h", "tags"]
        for i, field in enumerate(field_order):
            ttk.Label(top_row, text=field).grid(row=i, column=0, sticky="w", pady=2)
            ent = ttk.Entry(top_row, width=18)
            ent.configure(exportselection=False)
            ent.grid(row=i, column=1, sticky="ew", pady=2)
            ent.bind("<KeyRelease>", lambda _e, f=field: self._on_form_changed(source=f))
            ent.bind("<FocusOut>", lambda _e, f=field: self._on_form_changed(source=f))
            ent.bind("<Control-z>", self._on_entry_undo)
            ent.bind("<Control-y>", self._on_entry_redo)
            ent.bind("<Control-Z>", self._on_entry_undo)
            ent.bind("<Control-Y>", self._on_entry_redo)
            self._fields[field] = ent
            self._entry_field_map[ent] = field
            self._reset_entry_history(ent)

        group_row = len(field_order)
        ttk.Label(top_row, text="group").grid(row=group_row, column=0, sticky="w", pady=2)
        self._group_var = tk.StringVar(value="")
        self._group_combo = ttk.Combobox(top_row, textvariable=self._group_var, state="readonly")
        self._group_combo.grid(row=group_row, column=1, sticky="ew", pady=2)
        self._group_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed(source="group"))

        ttk.Label(desc_row, text="description").grid(row=0, column=0, sticky="w")
        self._desc_text = tk.Text(desc_row, height=6, wrap="word", exportselection=False)
        self._desc_text.configure(undo=True, autoseparators=True, maxundo=200)
        self._desc_text.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        self._desc_text.bind("<KeyRelease>", lambda _e: self._on_form_changed(source="description"))
        self._desc_text.bind("<FocusOut>", lambda _e: self._on_form_changed(source="description"))
        self._desc_text.bind("<Control-z>", self._on_desc_undo)
        self._desc_text.bind("<Control-y>", self._on_desc_redo)
        self._desc_text.bind("<Control-Z>", self._on_desc_undo)
        self._desc_text.bind("<Control-Y>", self._on_desc_redo)

        # Anchor-only: link ROIs that should be checked after using this anchor.
        self._linked_frame = ttk.LabelFrame(top_row, text="Linked ROIs (anchor only)", padding=8)
        linked_row = len(field_order) + 1
        self._linked_frame.grid(row=linked_row, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self._linked_frame.columnconfigure(0, weight=1)
        self._linked_frame.rowconfigure(1, weight=1)
        top_row.rowconfigure(linked_row, weight=1)

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

    def _refresh_list(self, *, preserve_form: bool = False) -> None:
        forced_rois = self._apply_forced_roi_activation()
        prev_keys = self._selected_keys()
        self.items_list.delete(0, tk.END)
        self._list_index_map = []

        self._refresh_filter_tags()

        group_names = {g.name for g in self.groups if g.name}

        def parent_key(name: str) -> str:
            if name in group_names:
                return name
            return ""

        groups_by_parent: dict[str, list[int]] = {}
        for idx, g in enumerate(self.groups):
            groups_by_parent.setdefault(parent_key(g.group), []).append(idx)

        items_by_parent: dict[str, list[tuple[ItemKind, int]]] = {}
        for idx, r in enumerate(self.rois):
            items_by_parent.setdefault(parent_key(r.group), []).append(("roi", idx))
        for idx, a in enumerate(self.anchors):
            items_by_parent.setdefault(parent_key(a.group), []).append(("anchor", idx))

        def subtree_visible(group_idx: int) -> bool:
            g = self.groups[group_idx]
            if self._item_matches_filter("group", g):
                return True
            for child_g in groups_by_parent.get(g.name, []):
                if subtree_visible(child_g):
                    return True
            for kind, item_idx in items_by_parent.get(g.name, []):
                item = self._item_for_kind(kind, item_idx)
                if self._item_matches_filter(kind, item):
                    return True
            return False

        def add_items(parent_name: str, depth: int) -> None:
            for group_idx in groups_by_parent.get(parent_name, []):
                if not subtree_visible(group_idx):
                    continue
                g = self.groups[group_idx]
                self._insert_list_item(kind="group", idx=group_idx, depth=depth, forced_rois=forced_rois)
                if not g.collapsed:
                    add_items(g.name, depth + 1)
            for kind, item_idx in items_by_parent.get(parent_name, []):
                item = self._item_for_kind(kind, item_idx)
                if not self._item_matches_filter(kind, item):
                    continue
                self._insert_list_item(kind=kind, idx=item_idx, depth=depth, forced_rois=forced_rois)

        add_items("", 0)
        self._refresh_roi_options()

        if prev_keys:
            self._select_list_keys(prev_keys)
            if not preserve_form:
                self._on_select(populate_form=True)
            return
        self.items_list.selection_clear(0, tk.END)
        self._on_select(populate_form=True)

    def _format_item_label(self, *, kind: ItemKind, name: str, active: bool, depth: int = 0) -> str:
        box = "[x]" if active else "[ ]"
        if kind == "roi":
            tag = "[ROI]"
        elif kind == "anchor":
            tag = "[ANCHOR]"
        else:
            tag = "[GROUP]"
        indent = "  " * max(0, depth)
        return f"{indent}{box} {tag} {name}"

    def _insert_list_item(self, *, kind: ItemKind, idx: int, depth: int, forced_rois: set[str]) -> None:
        item = self._item_for_kind(kind, idx)
        list_idx = self.items_list.size()
        self.items_list.insert(
            tk.END,
            self._format_item_label(kind=kind, name=item.name, active=item.active, depth=depth),
        )
        self._list_index_map.append((kind, idx))
        if kind == "roi" and item.name in forced_rois:
            self.items_list.itemconfig(list_idx, foreground="#808080")
        else:
            self.items_list.itemconfig(list_idx, foreground=self._items_default_fg)

    def _item_for_kind(self, kind: ItemKind, idx: int) -> Any:
        if kind == "roi":
            return self.rois[idx]
        if kind == "anchor":
            return self.anchors[idx]
        return self.groups[idx]

    def _selected_key(self) -> Optional[tuple[ItemKind, int]]:
        kind, idx = self._selected()
        if kind is None or idx is None:
            return None
        return (kind, idx)

    def _selected_keys(self) -> list[tuple[ItemKind, int]]:
        return list(self._selected_items())

    def _select_list_key(self, key: tuple[ItemKind, int]) -> bool:
        for list_idx, (kind, idx) in enumerate(self._list_index_map):
            if kind == key[0] and idx == key[1]:
                self.items_list.selection_clear(0, tk.END)
                self.items_list.selection_set(list_idx)
                return True
        return False

    def _select_list_keys(self, keys: list[tuple[ItemKind, int]]) -> None:
        if not keys:
            return
        want = {(k, i) for k, i in keys}
        self.items_list.selection_clear(0, tk.END)
        for list_idx, item in enumerate(self._list_index_map):
            if item in want:
                self.items_list.selection_set(list_idx)

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
        if list_idx < 0 or list_idx >= len(self._list_index_map):
            return ("roi", 0)
        return self._list_index_map[list_idx]

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

    def _split_tags(self, value: str) -> list[str]:
        return [t.strip() for t in (value or "").split(",") if t.strip()]

    def _refresh_filter_tags(self) -> None:
        if self._tag_frame_inner is None:
            return
        counts: dict[str, int] = {}
        for r in self.rois:
            for tag in set(self._split_tags(r.tags)):
                counts[tag] = counts.get(tag, 0) + 1
        for a in self.anchors:
            for tag in set(self._split_tags(a.tags)):
                counts[tag] = counts.get(tag, 0) + 1
        for g in self.groups:
            for tag in set(self._split_tags(g.tags)):
                counts[tag] = counts.get(tag, 0) + 1

        tags_sorted = sorted(counts.keys(), key=lambda t: (-counts.get(t, 0), str(t).lower()))
        new_vars: dict[str, tk.BooleanVar] = {}
        for tag in tags_sorted:
            var = self._filter_tag_vars.get(tag)
            if var is None:
                var = tk.BooleanVar(value=False)
            new_vars[tag] = var
        self._filter_tag_vars = new_vars

        for child in self._tag_frame_inner.winfo_children():
            child.destroy()
        if not tags_sorted:
            ttk.Label(self._tag_frame_inner, text="(no tags)").grid(row=0, column=0, sticky="w")
            return
        for i, tag in enumerate(tags_sorted):
            var = self._filter_tag_vars[tag]
            ttk.Checkbutton(self._tag_frame_inner, text=tag, variable=var, command=self._refresh_list).grid(
                row=i, column=0, sticky="w"
            )

    def _on_tag_frame_configure(self, _e: tk.Event) -> None:
        if self._tag_canvas is None:
            return
        try:
            self._tag_canvas.configure(scrollregion=self._tag_canvas.bbox("all"))
        except Exception:
            pass

    def _on_tag_canvas_configure(self, _e: tk.Event) -> None:
        if self._tag_canvas is None or self._tag_window is None:
            return
        try:
            self._tag_canvas.itemconfigure(self._tag_window, width=self._tag_canvas.winfo_width())
        except Exception:
            pass

    def _toggle_filter_logic(self) -> None:
        if self._filter_logic_var is None:
            return
        self._filter_logic_var.set("OR" if self._filter_logic_var.get() == "AND" else "AND")
        self._refresh_list()

    def _clear_filters(self) -> None:
        if self._filter_keyword_var is not None:
            self._filter_keyword_var.set("")
        for var in self._filter_tag_vars.values():
            var.set(False)
        self._refresh_list()

    def _filter_terms(self) -> list[str]:
        if self._filter_keyword_var is None:
            return []
        raw = self._filter_keyword_var.get() or ""
        return [t.strip().lower() for t in raw.split(",") if t.strip()]

    def _item_matches_filter(self, kind: ItemKind, item: Any) -> bool:
        selected_tags = [t.lower() for t, v in self._filter_tag_vars.items() if v.get()]
        if selected_tags:
            item_tags = [t.lower() for t in self._split_tags(getattr(item, "tags", ""))]
            tag_match = any(t in item_tags for t in selected_tags)
        else:
            tag_match = True

        terms = self._filter_terms()
        if not terms:
            keyword_match = True
        else:
            logic = "AND"
            if self._filter_logic_var is not None:
                logic = self._filter_logic_var.get() or "AND"
            keyword_match = self._match_terms(terms, kind, item, logic=logic)

        return tag_match and keyword_match

    def _match_terms(self, terms: list[str], kind: ItemKind, item: Any, *, logic: str) -> bool:
        fields: list[str] = []
        name = getattr(item, "name", "")
        desc = getattr(item, "description", "")
        if name:
            fields.append(str(name))
        if desc:
            fields.append(str(desc))
        if kind == "anchor":
            linked = getattr(item, "linked_rois", []) or []
            fields.extend(str(x) for x in linked if str(x).strip())
        fields_l = [f.lower() for f in fields]

        def term_matches(term: str) -> bool:
            return any(term in f for f in fields_l)

        if logic == "AND":
            return all(term_matches(t) for t in terms)
        return any(term_matches(t) for t in terms)

    def _group_descendants(self, name: str) -> set[str]:
        out: set[str] = set()
        pending = [name]
        while pending:
            cur = pending.pop()
            for g in self.groups:
                if g.group == cur and g.name and g.name not in out:
                    out.add(g.name)
                    pending.append(g.name)
        return out

    def _group_display_options(self, *, exclude: set[str]) -> list[tuple[str, str]]:
        group_names = {g.name for g in self.groups if g.name}
        groups_by_parent: dict[str, list[GroupDraft]] = {}
        for g in self.groups:
            if not g.name or g.name in exclude:
                continue
            parent = g.group if g.group in group_names and g.group not in exclude else ""
            groups_by_parent.setdefault(parent, []).append(g)

        def sort_key(g: GroupDraft) -> tuple[str, str]:
            return (g.name.lower(), g.name)

        out: list[tuple[str, str]] = []

        def add(parent: str, depth: int) -> None:
            for g in sorted(groups_by_parent.get(parent, []), key=sort_key):
                display = f"{'  ' * depth}{g.name}"
                out.append((display, g.name))
                add(g.name, depth + 1)

        add("", 0)
        return out

    def _set_group_active(self, group_name: str, active: bool) -> None:
        for g in self.groups:
            if g.group == group_name:
                g.active = active
                self._set_group_active(g.name, active)
        for r in self.rois:
            if r.group == group_name:
                r.active = active
        for a in self.anchors:
            if a.group == group_name:
                a.active = active

    def _refresh_group_options(
        self, *, kind: Optional[ItemKind] = None, idx: Optional[int] = None, set_value: bool = True
    ) -> None:
        if self._group_combo is None or self._group_var is None:
            return
        exclude: set[str] = set()
        current = ""
        if kind == "group" and idx is not None:
            current = self.groups[idx].group
            exclude.add(self.groups[idx].name)
            exclude.update(self._group_descendants(self.groups[idx].name))
        elif kind in {"roi", "anchor"} and idx is not None:
            current = self._item_for_kind(kind, idx).group
        options_pairs = self._group_display_options(exclude=exclude)
        self._group_display_to_name = {display: name for display, name in options_pairs}
        self._group_name_to_display = {name: display for display, name in options_pairs}
        options = ["(none)"] + [display for display, _name in options_pairs]
        try:
            self._group_combo["values"] = options
        except Exception:
            pass
        if set_value:
            display = self._group_name_to_display.get(current, "(none)") if current else "(none)"
            if display not in options:
                display = "(none)"
            self._group_var.set(display)

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
        self._on_form_canvas_configure(_e)

    def _on_form_canvas_configure(self, _e: tk.Event) -> None:
        if self._form_canvas is None or self._form_window is None:
            return
        try:
            canvas_w = self._form_canvas.winfo_width()
            canvas_h = self._form_canvas.winfo_height()
            req_h = self._form_inner.winfo_reqheight() if self._form_inner is not None else canvas_h
            height = max(req_h, canvas_h)
            self._form_canvas.itemconfigure(self._form_window, width=canvas_w, height=height)
            if self._form_inner is not None:
                self._form_inner.configure(width=canvas_w, height=height)
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
        delta = getattr(e, "delta", 0)
        if not delta:
            return "break"
        self.canvas.yview_scroll(int(-1 * (delta / 120)), "units")
        return "break"

    def _on_preview_shift_mousewheel(self, e: tk.Event) -> str:
        if self._screen_img is None:
            return "break"
        delta = getattr(e, "delta", 0)
        if not delta:
            return "break"
        self.canvas.xview_scroll(int(-1 * (delta / 120)), "units")
        return "break"

    def _on_preview_alt_mousewheel(self, e: tk.Event) -> str:
        if self._screen_img is None:
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

    def _selected_items(self) -> list[tuple[ItemKind, int]]:
        sel = self.items_list.curselection()
        out: list[tuple[ItemKind, int]] = []
        for idx in sel:
            if 0 <= idx < len(self._list_index_map):
                out.append(self._list_index_map[idx])
        return out

    def _selected(self) -> tuple[Optional[ItemKind], Optional[int]]:
        items = self._selected_items()
        if not items:
            return (None, None)
        kind, idx = items[0]
        return (kind, idx)

    def _on_items_click(self, e: tk.Event) -> Optional[str]:
        idx = self.items_list.nearest(e.y)
        if idx < 0:
            return None
        if idx >= len(self._list_index_map):
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
        elif kind == "group":
            item = self.groups[item_idx]
            item.active = not item.active
            self._set_group_active(item.name, item.active)
        else:
            item = self.anchors[item_idx]
            item.active = not item.active

        self._refresh_list()
        self.items_list.selection_clear(0, tk.END)
        self.items_list.selection_set(idx)
        self._on_select(populate_form=True)
        return "break"

    def _on_items_double_click(self, e: tk.Event) -> Optional[str]:
        idx = self.items_list.nearest(e.y)
        if idx < 0 or idx >= len(self._list_index_map):
            return None
        kind, item_idx = self._list_index_map[idx]
        if kind != "group":
            return None
        self.groups[item_idx].collapsed = not self.groups[item_idx].collapsed
        self._refresh_list(preserve_form=True)
        self._select_list_key(("group", item_idx))
        self._on_select(populate_form=True)
        return "break"

    def _on_select(self, *, populate_form: bool = True) -> None:
        sel_items = self._selected_items()
        self._clear_overlay()
        if not sel_items:
            if populate_form:
                self.kind_var.set("")
                for e in self._fields.values():
                    e.delete(0, tk.END)
                self._set_desc_state(True)
                self._set_description("")
                self._set_desc_state(False)
                if self._group_var is not None:
                    self._group_var.set("(none)")
                self._set_group_combo_state(False)
                self._set_linked_controls_enabled(False)
                self._set_linked_rois_ui([])
            return
        if len(sel_items) > 1:
            if not populate_form:
                return
            self._suppress_form_events = True
            self.kind_var.set(f"Multiple items ({len(sel_items)})")
            for key in ["name", "x", "y", "w", "h", "tags"]:
                self._set_field_state(key, False)
                self._set_field(key, "")
            self._set_desc_state(True)
            self._set_description("")
            self._set_desc_state(False)
            self._set_linked_controls_enabled(False)
            self._set_linked_rois_ui([])
            self._refresh_group_options(set_value=False)
            self._set_group_combo_state(True)
            groups = []
            for kind, idx in sel_items:
                item = self._item_for_kind(kind, idx)
                groups.append(item.group or "")
            group_value = groups[0] if groups and all(g == groups[0] for g in groups) else "(mixed)"
            if self._group_var is not None:
                if group_value in {"", "(none)"}:
                    self._group_var.set("(none)")
                else:
                    display = self._group_name_to_display.get(group_value, group_value) if group_value != "(mixed)" else "(mixed)"
                    self._group_var.set(display)
            self._suppress_form_events = False
            return

        kind, idx = sel_items[0]

        if populate_form:
            self._suppress_form_events = True
            if kind == "roi":
                item = self.rois[idx]
                self.kind_var.set("ROI (Observation)")
                self._set_group_combo_state(True)
                self._set_desc_state(True)
                self._set_field_state("name", True)
                self._set_field_state("x", True)
                self._set_field_state("y", True)
                self._set_field_state("w", True)
                self._set_field_state("h", True)
                self._set_field_state("tags", True)
                self._set_field("name", item.name)
                self._set_field("x", str(item.x))
                self._set_field("y", str(item.y))
                self._set_field("w", str(item.w))
                self._set_field("h", str(item.h))
                self._set_field("description", item.description)
                self._set_field("tags", item.tags)
                self._refresh_group_options(kind="roi", idx=idx)
                self._set_linked_controls_enabled(False)
                self._set_linked_rois_ui([])
            elif kind == "anchor":
                item = self.anchors[idx]
                self.kind_var.set("Anchor (Action click point)")
                self._set_group_combo_state(True)
                self._set_desc_state(True)
                self._set_field_state("name", True)
                self._set_field_state("x", True)
                self._set_field_state("y", True)
                self._set_field_state("w", False)
                self._set_field_state("h", False)
                self._set_field_state("tags", True)
                self._set_field("name", item.name)
                self._set_field("x", str(item.x))
                self._set_field("y", str(item.y))
                self._set_field("w", "")
                self._set_field("h", "")
                self._set_field("description", item.description)
                self._set_field("tags", item.tags)
                self._refresh_group_options(kind="anchor", idx=idx)
                self._set_linked_controls_enabled(True)
                self._set_linked_rois_ui(list(item.linked_rois or []))
            else:
                item = self.groups[idx]
                self.kind_var.set("Group (Folder)")
                self._set_group_combo_state(True)
                self._set_desc_state(True)
                self._set_field_state("name", True)
                self._set_field_state("x", False)
                self._set_field_state("y", False)
                self._set_field_state("w", False)
                self._set_field_state("h", False)
                self._set_field_state("tags", True)
                self._set_field("name", item.name)
                self._set_field("x", "")
                self._set_field("y", "")
                self._set_field("w", "")
                self._set_field("h", "")
                self._set_field("description", item.description)
                self._set_field("tags", item.tags)
                self._refresh_group_options(kind="group", idx=idx)
                self._set_linked_controls_enabled(False)
                self._set_linked_rois_ui([])
            self._suppress_form_events = False

        if kind == "roi":
            self._draw_existing_roi(self.rois[idx])
        elif kind == "anchor":
            self._draw_existing_anchor(self.anchors[idx])

    def _set_field(self, key: str, value: str) -> None:
        if key == "description":
            self._set_description(value)
            return
        ent = self._fields[key]
        prev_state = str(ent.cget("state"))
        if prev_state == "disabled":
            ent.configure(state=tk.NORMAL)
        ent.delete(0, tk.END)
        ent.insert(0, value)
        if prev_state == "disabled":
            ent.configure(state=tk.DISABLED)
        self._reset_entry_history(ent)

    def _set_field_state(self, key: str, enabled: bool) -> None:
        ent = self._fields.get(key)
        if ent is None:
            return
        ent.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _set_desc_state(self, enabled: bool) -> None:
        if self._desc_text is None:
            return
        try:
            self._desc_text.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        except Exception:
            pass

    def _set_group_combo_state(self, enabled: bool) -> None:
        if self._group_combo is None:
            return
        try:
            self._group_combo.configure(state="readonly" if enabled else "disabled")
        except Exception:
            pass

    def _reset_entry_history(self, ent: tk.Entry) -> None:
        self._entry_history[ent] = [(ent.get(), int(ent.index(tk.INSERT)))]
        self._entry_history_index[ent] = 0

    def _record_entry_history(self, ent: tk.Entry) -> None:
        if self._entry_history_lock:
            return
        text = ent.get()
        cursor = int(ent.index(tk.INSERT))
        hist = self._entry_history.get(ent)
        idx = self._entry_history_index.get(ent, -1)
        if not hist:
            self._entry_history[ent] = [(text, cursor)]
            self._entry_history_index[ent] = 0
            return
        if 0 <= idx < len(hist) and hist[idx][0] == text:
            hist[idx] = (text, cursor)
            return
        if idx + 1 < len(hist):
            hist = hist[: idx + 1]
        hist.append((text, cursor))
        self._entry_history[ent] = hist
        self._entry_history_index[ent] = len(hist) - 1

    def _apply_entry_state(self, ent: tk.Entry, state: tuple[str, int], idx: int) -> None:
        self._entry_history_lock = True
        try:
            ent.delete(0, tk.END)
            ent.insert(0, state[0])
            ent.icursor(state[1])
        finally:
            self._entry_history_lock = False
        self._entry_history_index[ent] = idx

    def _on_entry_undo(self, e: tk.Event) -> str:
        ent = e.widget
        hist = self._entry_history.get(ent)
        idx = self._entry_history_index.get(ent, 0)
        if not hist or idx <= 0:
            return "break"
        self._apply_entry_state(ent, hist[idx - 1], idx - 1)
        field = self._entry_field_map.get(ent)
        if field:
            self._on_form_changed(source=field)
        return "break"

    def _on_entry_redo(self, e: tk.Event) -> str:
        ent = e.widget
        hist = self._entry_history.get(ent)
        idx = self._entry_history_index.get(ent, 0)
        if not hist or idx + 1 >= len(hist):
            return "break"
        self._apply_entry_state(ent, hist[idx + 1], idx + 1)
        field = self._entry_field_map.get(ent)
        if field:
            self._on_form_changed(source=field)
        return "break"

    def _on_desc_undo(self, _e: tk.Event) -> str:
        if self._desc_text is None:
            return "break"
        try:
            self._desc_text.edit_undo()
        except Exception:
            pass
        self._on_form_changed(source="description")
        return "break"

    def _on_desc_redo(self, _e: tk.Event) -> str:
        if self._desc_text is None:
            return "break"
        try:
            self._desc_text.edit_redo()
        except Exception:
            pass
        self._on_form_changed(source="description")
        return "break"

    def _set_description(self, value: str) -> None:
        if self._desc_text is None:
            return
        self._desc_text.delete("1.0", tk.END)
        self._desc_text.insert("1.0", value or "")
        try:
            self._desc_text.edit_reset()
        except Exception:
            pass

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
        self._refresh_list(preserve_form=True)

    def _on_form_changed(self, *, source: str) -> None:
        if self._suppress_form_events:
            return
        selected = self._selected_items()
        if not selected:
            return
        if len(selected) > 1:
            if source != "group":
                return
            if self._group_var is None:
                return
            display = self._group_var.get().strip()
            if display in {"", "(none)", "(mixed)"}:
                group_in = ""
            else:
                group_in = self._group_display_to_name.get(display, display)
            group_names = {g.name for g in self.groups if g.name}
            if group_in and group_in not in group_names:
                return
            for kind, idx in selected:
                if kind != "group" or not group_in:
                    continue
                cur = self.groups[idx]
                if group_in == cur.name or group_in in self._group_descendants(cur.name):
                    messagebox.showinfo("Invalid group", "A group cannot be its own parent.")
                    self._refresh_group_options(set_value=False)
                    return
            for kind, idx in selected:
                if kind == "roi":
                    self.rois[idx].group = group_in
                elif kind == "anchor":
                    self.anchors[idx].group = group_in
                else:
                    self.groups[idx].group = group_in
            self._refresh_list(preserve_form=True)
            self._select_list_keys(selected)
            self._on_select(populate_form=True)
            return

        kind, idx = selected[0]

        if source in self._fields and source != "description":
            self._record_entry_history(self._fields[source])

        name_in = self._fields["name"].get()
        x_in = self._try_int(self._fields["x"].get())
        y_in = self._try_int(self._fields["y"].get())
        description_in = self._get_description()
        tags_in = self._fields["tags"].get()
        display = ""
        if self._group_var is not None:
            display = self._group_var.get().strip()
        if display in {"", "(none)", "(mixed)"}:
            group_in = ""
        else:
            group_in = self._group_display_to_name.get(display, display)
        group_names = {g.name for g in self.groups if g.name}
        if group_in and group_in not in group_names:
            group_in = ""
            if self._group_var is not None:
                self._group_var.set("(none)")

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
                group=group_in if group_in in group_names else "",
            )
            if source in {"name", "group", "tags", "description"}:
                self._refresh_roi_options()
                self._update_selected_list_label(kind="roi", idx=idx)
            if source in {"x", "y", "w", "h"}:
                self._clear_overlay()
                self._draw_existing_roi(self.rois[idx])
        else:
            if kind == "anchor":
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
                    group=group_in if group_in in group_names else "",
                )
                if source in {"name", "group", "tags", "description"}:
                    self._update_selected_list_label(kind="anchor", idx=idx)
                if source in {"x", "y"}:
                    self._clear_overlay()
                    self._draw_existing_anchor(self.anchors[idx])
                if source in {"linked_rois"}:
                    self._refresh_list()
                    self._select_list_key(("anchor", idx))
            else:
                cur = self.groups[idx]
                if source == "group" and group_in:
                    if group_in == cur.name or group_in in self._group_descendants(cur.name):
                        messagebox.showinfo("Invalid group", "A group cannot be its own parent.")
                        self._refresh_group_options(kind="group", idx=idx)
                        return
                new_name = name_in or cur.name
                self.groups[idx] = GroupDraft(
                    name=new_name,
                    description=description_in,
                    tags=tags_in,
                    active=cur.active,
                    group=group_in if group_in in group_names else "",
                )
                if source == "name" and new_name != cur.name:
                    for g in self.groups:
                        if g.group == cur.name:
                            g.group = new_name
                    for r in self.rois:
                        if r.group == cur.name:
                            r.group = new_name
                    for a in self.anchors:
                        if a.group == cur.name:
                            a.group = new_name
                if source in {"name", "group", "tags", "description"}:
                    self._update_selected_list_label(kind="group", idx=idx)

    def _add_roi(self) -> None:
        existing = {r.name for r in self.rois} | {a.name for a in self.anchors} | {g.name for g in self.groups}
        name = _dedupe_name(existing, "new_roi")
        parent = ""
        kind, idx = self._selected()
        if kind == "group" and idx is not None:
            parent = self.groups[idx].name
        self.rois.append(RoiDraft(name=name, description="", group=parent))
        self._refresh_list()
        self._select_list_key(("roi", len(self.rois) - 1))
        self._on_select(populate_form=True)

    def _add_anchor(self) -> None:
        existing = {r.name for r in self.rois} | {a.name for a in self.anchors} | {g.name for g in self.groups}
        name = _dedupe_name(existing, "new_anchor")
        parent = ""
        kind, idx = self._selected()
        if kind == "group" and idx is not None:
            parent = self.groups[idx].name
        self.anchors.append(AnchorDraft(name=name, description="", group=parent))
        self._refresh_list()
        self._select_list_key(("anchor", len(self.anchors) - 1))
        self._on_select(populate_form=True)

    def _add_group(self) -> None:
        existing = {r.name for r in self.rois} | {a.name for a in self.anchors} | {g.name for g in self.groups}
        name = _dedupe_name(existing, "new_group")
        parent = ""
        kind, idx = self._selected()
        if kind == "group" and idx is not None:
            parent = self.groups[idx].name
        self.groups.append(GroupDraft(name=name, description="", group=parent))
        self._refresh_list()
        self._select_list_key(("group", len(self.groups) - 1))
        self._on_select(populate_form=True)

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
        elif kind == "anchor":
            self.anchors.pop(idx)
        else:
            removed = self.groups.pop(idx)
            removed_name = (removed.name or "").strip()
            parent = removed.group
            for g in self.groups:
                if g.group == removed_name:
                    g.group = parent
            for r in self.rois:
                if r.group == removed_name:
                    r.group = parent
            for a in self.anchors:
                if a.group == removed_name:
                    a.group = parent
        self._refresh_list()
        self._on_select()

    def _load_workspace_dialog(self) -> None:
        initial_dir = self.workspaces_dir if self.workspaces_dir.exists() else self.workspace_path.parent
        path = filedialog.askopenfilename(
            title="Load workspace",
            initialdir=str(initial_dir),
            filetypes=[("Workspace JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_workspace(Path(path))

    def _export_workspace_dialog(self) -> None:
        initial_dir = self.workspaces_dir if self.workspaces_dir.exists() else self.workspace_path.parent
        default_name = self.workspace_path.name if self.workspace_path else "workspace.json"
        path = filedialog.asksaveasfilename(
            title="Export workspace",
            initialdir=str(initial_dir),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("Workspace JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self._write_workspace(Path(path), action="Exported", update_path=False)

    def _settings_workspace_value(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.repo_root))
        except Exception:
            return str(path)

    def _update_agent_workspace_setting(self, path: Path) -> None:
        try:
            settings = load_tui_settings(self._settings_path)
            workspace_value = self._settings_workspace_value(path)
            merged = merge_settings(settings, {"workspace": workspace_value})
            save_tui_settings(merged, self._settings_path)
            messagebox.showinfo("Agent workspace", f"Agent workspace set to {workspace_value}")
        except Exception as e:
            messagebox.showerror("Agent workspace", f"Failed to update agent workspace: {e}")

    def _set_agent_workspace_dialog(self) -> None:
        initial_dir = self.workspaces_dir if self.workspaces_dir.exists() else self.workspace_path.parent
        path = filedialog.askopenfilename(
            title="Set agent workspace",
            initialdir=str(initial_dir),
            filetypes=[("Workspace JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self._update_agent_workspace_setting(Path(path))

    def _set_agent_workspace_current(self) -> None:
        self._update_agent_workspace_setting(self.workspace_path)

    def _load_workspace(self, path: Path) -> None:
        try:
            raw = _load_workspace_raw(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self.workspace_path = path
        self.raw = raw
        self._load_workspace_data(raw)
        self.mode = "idle"
        self._drawing_item_index = None
        self._draw_start = None
        self._clear_overlay()
        self._refresh_list()
        self.items_list.selection_clear(0, tk.END)
        self._on_select(populate_form=True)
        self._status("idle")
        messagebox.showinfo("Workspace loaded", f"Loaded {path}")

    def _write_workspace(self, path: Path, *, action: str, update_path: bool) -> None:
        try:
            self._apply_forced_roi_activation()
            names: set[str] = set()
            roi_names: set[str] = set()
            for r in self.rois:
                if not r.name.strip():
                    raise ValueError("ROI name cannot be empty")
                if r.name in names:
                    raise ValueError(f"Duplicate name: {r.name!r}")
                names.add(r.name)
                roi_names.add(r.name)
                if r.w <= 0 or r.h <= 0:
                    raise ValueError(f"ROI {r.name!r} must have positive w/h")
            for a in self.anchors:
                if not a.name.strip():
                    raise ValueError("Anchor name cannot be empty")
                if a.name in names:
                    raise ValueError(f"Duplicate name: {a.name!r}")
                names.add(a.name)
                a.linked_rois = [x for x in (a.linked_rois or []) if x in roi_names]
            group_names: set[str] = set()
            for g in self.groups:
                if not g.name.strip():
                    raise ValueError("Group name cannot be empty")
                if g.name in names:
                    raise ValueError(f"Duplicate name: {g.name!r}")
                names.add(g.name)
                group_names.add(g.name)

            for g in self.groups:
                if g.group and g.group not in group_names:
                    g.group = ""
            for r in self.rois:
                if r.group and r.group not in group_names:
                    r.group = ""
            for a in self.anchors:
                if a.group and a.group not in group_names:
                    a.group = ""

            for g in self.groups:
                seen: set[str] = set()
                cur = g
                while cur.group:
                    if cur.group in seen:
                        raise ValueError(f"Group cycle detected at {cur.name!r}")
                    seen.add(cur.group)
                    parent = next((x for x in self.groups if x.name == cur.group), None)
                    if parent is None:
                        break
                    cur = parent

            out = dict(self.raw)
            out["rois"] = [r.to_json() for r in self.rois]
            out["anchors"] = [a.to_json() for a in self.anchors]
            out["groups"] = [g.to_json() for g in self.groups]
            path.write_text(json.dumps(out, indent=2, sort_keys=False), encoding="utf-8")
            if update_path:
                self.workspace_path = path
            messagebox.showinfo(action, f"{action} to {path}")
        except Exception as e:
            messagebox.showerror(f"{action} failed", str(e))

    def _save(self) -> None:
        self._write_workspace(self.workspace_path, action="Saved", update_path=True)

    def _begin_draw_roi(self) -> None:
        selected = self._selected_items()
        if len(selected) != 1 or selected[0][0] != "roi":
            messagebox.showinfo("Select ROI", "Select an ROI item first (or Add ROI).")
            return
        _, idx = selected[0]
        self.mode = "draw_roi"
        self._drawing_item_index = idx
        self._clear_overlay()
        self._status("Draw ROI: click+drag on screenshot")

    def _begin_pick_anchor(self) -> None:
        selected = self._selected_items()
        if len(selected) != 1 or selected[0][0] != "anchor":
            messagebox.showinfo("Select anchor", "Select an Anchor item first (or Add Anchor).")
            return
        _, idx = selected[0]
        self.mode = "pick_anchor"
        self._drawing_item_index = idx
        self._clear_overlay()
        self._status("Pick anchor: click on screenshot")

    def _status(self, text: str) -> None:
        name = self.workspace_path.name if self.workspace_path else "workspace.json"
        self.title(f"Workspace Calibrator - {name} - {text}")

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
