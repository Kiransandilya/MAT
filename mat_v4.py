#!/usr/bin/env python3
"""
MAT: Multi-Channel Annotation Tool v3.0
========================================

A professional-grade annotation tool for large scientific images with:
- Dynamic multi-channel architecture (add/remove channels per session)
- Memory-efficient processing for 6K+ images
- AI-powered mask propagation using optical flow
- Session persistence with recent sessions management
- Comprehensive image enhancement filters
- Self-contained sessions (includes original TIF)

Author: Kiran Sandilya
Version: 3.0
Date: 2025-12-18
License: MIT

Features:
- Dynamic Channels: Add/remove/edit channels on demand
- Propagation: Forward/backward mask propagation with confidence scoring
- Recent Sessions: Persistent recent sessions list (last 10)
- Enhanced Loading: Save TIF with session for portability
- 6 Enhancement Filters: Brightness, contrast, blur, filters, temporal smoothing
- Memory Optimization: LRU caching, automatic cleanup
- Professional UI: Dark theme, collapsible panels, keyboard shortcuts
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import os
import sys
import gc
import time
import datetime
import threading
import tempfile
import warnings
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum

# Third-party libraries
import numpy as np
import cv2
from PIL import Image, ImageTk

# GUI framework
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional: Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring disabled.")

# Optional: Advanced filtering
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using fallback for smoothing.")
    def savgol_filter(data, window_length, polyorder):
        """Fallback implementation when scipy unavailable."""
        return np.array(data)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class Config:
    """
    Centralized configuration for MAT.
    
    All configurable parameters are defined here for easy tuning.
    Values can be overridden via environment variables with prefix 'MAT_'.
    """
    
    # Application info
    APP_NAME = "MAT"
    APP_FULL_NAME = "Multi-Channel Annotation Tool"
    APP_VERSION = "3.0"
    APP_DESCRIPTION = "Professional-grade annotation tool for large scientific image stacks"

    # Key Features
    APP_FEATURES = [
        "Save Session - Save your current Progress"
        "Load Session - Load your saved Progress"
        "Multi-channel annotation with dynamic channel management",
        "Memory-efficient processing for 6K+ images with intelligent caching",
        "AI-powered mask propagation using optical flow",
        "Session persistence - save and resume work anytime",
        "Real-time image enhancement filters (brightness, contrast, blur, etc.)",
        "Contour drawing tools with fill/outline modes",
        "Undo/redo support for drawing operations",
        "Automatic frame playback with adjustable FPS",
        "Export to multi-folder structure (Image, Overlay, Masks)",
        "Recent sessions management for quick access",
        "Self-contained sessions include original TIF for portability"
    ]

    # Performance settings
    MAX_DISPLAY_SIZE = int(os.getenv('MAT_MAX_DISPLAY_SIZE', '1536'))
    DOWNSCALE_THRESHOLD = int(os.getenv('MAT_DOWNSCALE_THRESHOLD', '2048'))
    # MAX_FRAMES_IN_MEMORY = int(os.getenv('MAT_MAX_FRAMES', '2'))
    # MEMORY_THRESHOLD_MB = int(os.getenv('MAT_MEMORY_THRESHOLD', '800'))
    # Frame caching - OPTIMIZED for 30+ frame workflow
    MAX_FRAMES_IN_MEMORY = int(os.getenv('MAT_MAX_FRAMES', '30'))        # 40→10
    PRELOAD_WINDOW_SIZE = int(os.getenv('MAT_PRELOAD_WINDOW', '5'))      # 15→5
    MEMORY_THRESHOLD_MB = int(os.getenv('MAT_MEMORY_THRESHOLD', '800'))  # 1200→800
    AGGRESSIVE_CACHING = False                                            # True→False
    CLEANUP_INTERVAL_MS = int(os.getenv('MAT_CLEANUP_INTERVAL', '10000'))

    # Playback settings
    DEFAULT_PLAYBACK_FPS = 10  # Default FPS for playback
    MIN_PLAYBACK_FPS = 1
    MAX_PLAYBACK_FPS = 60
    
    # Rendering optimization
    RESIZE_METHOD = cv2.INTER_LINEAR
    ZOOM_RESIZE_METHOD = cv2.INTER_NEAREST
    UPDATE_THROTTLE_MS = int(os.getenv('MAT_UPDATE_THROTTLE', '20'))
    DRAW_THROTTLE_MS = int(os.getenv('MAT_DRAW_THROTTLE', '10'))
    
    # Zoom limits
    ZOOM_MIN = 0.1
    ZOOM_MAX = 10.0
    ZOOM_STEP = 1.2
    
    # UI settings
    DEFAULT_WINDOW_WIDTH = 1600  # Bigger default
    DEFAULT_WINDOW_HEIGHT = 1000
    MIN_WINDOW_WIDTH = 1000  # Prevent toolbar squashing
    MIN_WINDOW_HEIGHT = 700
    
    # Theme colors (professional dark theme)
    COLOR_BG_PRIMARY = '#2b2b2b'
    COLOR_BG_SECONDARY = '#333333'
    COLOR_BG_CANVAS = '#222222'
    COLOR_FG_PRIMARY = '#ffffff'
    COLOR_FG_SECONDARY = '#cccccc'
    COLOR_ACCENT = '#4CAF50'
    COLOR_ERROR = '#F44336'
    COLOR_WARNING = '#FFC107'
    COLOR_INFO = '#2196F3'
    
    # Brush settings
    BRUSH_SIZE_MIN = 0.5
    BRUSH_SIZE_MAX = 25.0
    BRUSH_SIZE_DEFAULT = 5.0
    
    # Undo settings
    UNDO_STACK_SIZE = 5
    
    # Channel settings
    CHANNEL_OPACITY = 0.4
    DEFAULT_CHANNELS = [
        ("High Confidence", (0, 255, 0), "🟢"),
        ("Medium Confidence", (255, 165, 0), "🟠"),
        ("Low Confidence", (255, 0, 0), "🔴"),
        ("Borderline Cases", (0, 0, 255), "🔵"),
    ]
    
    # Optical flow settings
    OPTICAL_FLOW_METHOD = 'dual_tv_l1'
    FLOW_PYR_SCALE = 0.5
    FLOW_LEVELS = 3
    FLOW_WINSIZE = 15
    FLOW_ITERATIONS = 3
    FLOW_POLY_N = 5
    FLOW_POLY_SIGMA = 1.2
    
    # File I/O settings
    SUPPORTED_FORMATS = ('.tif', '.tiff')
    EXPORT_COMPRESSION = 'tiff_lzw'
    FRAME_DIGIT_PADDING = 4
    
    # Folder structure
    SUBFOLDER_IMAGES = 'Image'
    SUBFOLDER_OVERLAYS = 'Overlay'
    SUBFOLDER_MASKS = 'Masks'  # NEW: Parent folder for all masks
    SUBFOLDER_MASK_ALL = 'Mask_All'  # NEW: Combined masks folder
    
    # Session files
    SESSION_LOG_BINARY = 'session.dat'
    SESSION_LOG_CSV = 'session.csv'
    SESSION_METADATA = 'session_info.json'
    ORIGINAL_TIF_NAME = 'original.tif'
    RECENT_SESSIONS_FILE = 'recent_sessions.json'
    
    # Recent sessions
    MAX_RECENT_SESSIONS = 10
    
    # Auto-save
    AUTOSAVE_ON_FRAME_SAVE = True
    AUTOSAVE_ON_EXIT = True
    AUTO_SAVE_INTERVAL_MIN = 5
    
    # Export options
    EXPORT_FULL_RESOLUTION = True
    EXPORT_TIF_STACKS = True
    CREATE_OVERLAYS = True
    
    # Debug
    DEBUG_MODE = os.getenv('MAT_DEBUG', 'false').lower() == 'true'
    ENABLE_PROFILING = os.getenv('MAT_PROFILE', 'false').lower() == 'true'
    LOG_TO_FILE = False
    LOG_FILE_PATH = 'mat_annotation.log'


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

VERSION = "3.0"
BUILD_DATE = "2025-12-18"
CURRENT_SESSION = None
PERFORMANCE_STATS = {
    'frame_loads': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'memory_cleanups': 0,
    'total_draw_operations': 0,
}


# =============================================================================
# ENUMS
# =============================================================================

class FrameIndexing(Enum):
    """Frame numbering convention in source TIF."""
    ZERO_BASED = "zero_based"
    ONE_BASED = "one_based"


class ToolType(Enum):
    """Drawing tool types."""
    DRAW = "draw"
    ERASE = "erase"
    ERASE_ADMIN = "erase_admin"
    CONTOUR = "contour" 

class ResampleMethod(Enum):
    """Image resampling methods."""
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    LANCZOS = cv2.INTER_LANCZOS4


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_timestamp() -> str:
    """Get current timestamp in standard format."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(message: str, level: str = 'INFO'):
    """Simple logging function."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    
    if Config.LOG_TO_FILE:
        try:
            with open(Config.LOG_FILE_PATH, 'a') as f:
                f.write(f"[{timestamp}] [{level}] {message}\n")
        except:
            pass


def debug_log(message: str):
    """Log debug messages (only if debug mode enabled)."""
    if Config.DEBUG_MODE:
        log(message, 'DEBUG')


def profile_function(func):
    """Decorator for profiling function execution time."""
    def wrapper(*args, **kwargs):
        if Config.ENABLE_PROFILING:
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            log(f"{func.__name__} took {elapsed*1000:.2f}ms", 'PROFILE')
            return result
        return func(*args, **kwargs)
    return wrapper


# =============================================================================
# CHANNEL DEFINITION (Dynamic)
# =============================================================================

@dataclass
class ChannelDefinition:
    """
    Definition of an annotation channel (dynamic).
    
    Supports runtime creation/modification of channels.
    """
    
    id: str
    """Unique identifier (e.g., 'channel_001')"""
    
    name: str
    """Display name (e.g., 'High Confidence')"""
    
    color_rgb: Tuple[int, int, int]
    """RGB color (0-255)"""
    
    emoji: str = "⚫"
    """Optional emoji indicator"""
    
    folder_name: str = ""
    """Folder name for export (auto-generated if empty)"""
    
    visible: bool = True
    """Visibility in overlay"""
    
    order: int = 0
    """Display/overlay priority"""
    
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    """Creation timestamp"""
    
    def __post_init__(self):
        """Generate folder name if not provided."""
        if not self.folder_name:
            self.folder_name = f"Mask_{self.name.replace(' ', '_')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'color_rgb': list(self.color_rgb),
            'emoji': self.emoji,
            'folder_name': self.folder_name,
            'visible': self.visible,
            'order': self.order,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelDefinition':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            color_rgb=tuple(data['color_rgb']),
            emoji=data.get('emoji', '⚫'),
            folder_name=data.get('folder_name', ''),
            visible=data.get('visible', True),
            order=data.get('order', 0),
            created_at=datetime.datetime.fromisoformat(data['created_at'])
        )


# =============================================================================
# CHANNEL MANAGER
# =============================================================================

class ChannelManager:
    """
    Manages dynamic annotation channels for a session.
    
    Features:
    - Add/remove/edit channels
    - Persist channel definitions
    - Validate uniqueness
    - Provide default channels
    """
    
    def __init__(self):
        self.channels: Dict[str, ChannelDefinition] = {}
        self._next_id = 1
    
    def add_channel(self, name: str, color_rgb: Tuple[int, int, int],
                    emoji: str = "⚫") -> ChannelDefinition:
        """Add new channel."""
        channel_id = f"channel_{self._next_id:03d}"
        self._next_id += 1
        
        channel = ChannelDefinition(
            id=channel_id,
            name=name,
            color_rgb=color_rgb,
            emoji=emoji,
            order=len(self.channels)
        )
        
        self.channels[channel_id] = channel
        log(f"Added channel: {name}")
        return channel
    
    def remove_channel(self, channel_id: str) -> bool:
        """Remove channel by ID."""
        if channel_id in self.channels:
            name = self.channels[channel_id].name
            del self.channels[channel_id]
            log(f"Removed channel: {name}")
            return True
        return False
    
    def update_channel(self, channel_id: str, **kwargs) -> bool:
        """Update channel properties."""
        if channel_id not in self.channels:
            return False
        
        channel = self.channels[channel_id]
        
        if 'name' in kwargs:
            channel.name = kwargs['name']
        if 'color_rgb' in kwargs:
            channel.color_rgb = kwargs['color_rgb']
        if 'emoji' in kwargs:
            channel.emoji = kwargs['emoji']
        if 'visible' in kwargs:
            channel.visible = kwargs['visible']
        if 'order' in kwargs:
            channel.order = kwargs['order']
        
        return True
    
    def get_channel(self, channel_id: str) -> Optional[ChannelDefinition]:
        """Get channel by ID."""
        return self.channels.get(channel_id)
    
    def get_all_channels(self, sorted_by_order: bool = True) -> List[ChannelDefinition]:
        """Get all channels, optionally sorted by order."""
        channels = list(self.channels.values())
        if sorted_by_order:
            channels.sort(key=lambda c: c.order)
        return channels
    
    def get_visible_channels(self) -> List[ChannelDefinition]:
        """Get only visible channels."""
        return [ch for ch in self.get_all_channels() if ch.visible]
    
    def load_default_channels(self):
        """Load default 4-channel configuration."""
        for name, color, emoji in Config.DEFAULT_CHANNELS:
            self.add_channel(name, color, emoji)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize all channels."""
        return {
            'next_id': self._next_id,
            'channels': {cid: ch.to_dict() for cid, ch in self.channels.items()}
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Deserialize channels."""
        self._next_id = data.get('next_id', 1)
        self.channels = {}
        
        for cid, ch_data in data.get('channels', {}).items():
            self.channels[cid] = ChannelDefinition.from_dict(ch_data)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ImageMetadata:
    """Comprehensive metadata for loaded image stack."""
    
    file_path: Path
    total_frames: int
    original_width: int
    original_height: int
    display_width: int
    display_height: int
    scale_factor: float
    dtype: str
    frame_indexing: FrameIndexing = FrameIndexing.ZERO_BASED
    loaded_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    file_size_mb: float = 0.0
    
    @property
    def original_megapixels(self) -> float:
        return (self.original_width * self.original_height) / 1_000_000
    
    @property
    def display_megapixels(self) -> float:
        return (self.display_width * self.display_height) / 1_000_000
    
    @property
    def memory_savings_percent(self) -> float:
        if self.scale_factor >= 1.0:
            return 0.0
        return (1 - self.scale_factor ** 2) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': str(self.file_path),
            'total_frames': self.total_frames,
            'original_width': self.original_width,
            'original_height': self.original_height,
            'display_width': self.display_width,
            'display_height': self.display_height,
            'scale_factor': self.scale_factor,
            'dtype': self.dtype,
            'file_size_mb': self.file_size_mb
        }
        
    # def to_dict(self) -> Dict[str, Any]:
    #     return {
    #         'project_name': self.project_name,
    #         'created_at': self.created_at.isoformat(),
    #         'last_modified': self.last_modified.isoformat(),
    #         'image_metadata': self.image_metadata.to_dict(),
    #         'channels': self.channel_manager.to_dict(),
    #         'labeled_frames': self.labeled_frames,
    #         'active_channel_id': getattr(self, 'active_channel_id', None),  # FIXED
    #         'mask_opacity': getattr(self, 'mask_opacity', Config.CHANNEL_OPACITY),  # FIXED
    #         'statistics': {
    #             'total_labeled_frames': len(self.labeled_frames),
    #             'total_labeled_pixels': self.total_labeled_pixels,
    #         },
    #     }


@dataclass
class FrameAnnotation:
    """Multi-channel annotation data for a single frame."""
    
    frame_index: int
    
    # Dynamic channel masks (channel_id → mask array)
    channel_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Per-channel statistics
    channel_pixels: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    is_keyframe: bool = False
    propagated_from: Optional[int] = None
    edit_count: int = 0
    notes: str = ""
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def get_mask(self, channel_id: str) -> Optional[np.ndarray]:
        """Get mask for specific channel ID."""
        return self.channel_masks.get(channel_id)
    
    def set_mask(self, channel_id: str, mask: np.ndarray):
        """Set mask for specific channel ID."""
        self.channel_masks[channel_id] = mask
        self.channel_pixels[channel_id] = int(np.sum(mask > 0))
        self.timestamp = datetime.datetime.now()
        self.edit_count += 1
    
    def get_pixel_count(self, channel_id: str) -> int:
        """Get pixel count for specific channel."""
        return self.channel_pixels.get(channel_id, 0)
    
    @property
    def total_labeled_pixels(self) -> int:
        """Total pixels across all channels."""
        return sum(self.channel_pixels.values())
    
    @property
    def has_annotations(self) -> bool:
        """Check if any channel has annotations."""
        return self.total_labeled_pixels > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (excludes large mask arrays)."""
        return {
            'frame_index': self.frame_index,
            'channel_pixels': self.channel_pixels.copy(),
            'total_pixels': self.total_labeled_pixels,
            'timestamp': self.timestamp.isoformat(),
            'is_keyframe': self.is_keyframe,
            'propagated_from': self.propagated_from,
            'edit_count': self.edit_count,
            'notes': self.notes,
        }


@dataclass
class SessionState:
    """State of an annotation session - BACKWARD COMPATIBLE."""
    
    # Project info
    project_name: str = ""
    tif_path: str = ""
    project_dir: str = ""
    
    # Frame info
    current_frame_index: int = 0
    total_frames: int = 0
    
    # Image metadata (ADDED - was missing!)
    image_metadata: Optional[ImageMetadata] = None
    
    # Annotations
    annotations: Dict[int, FrameAnnotation] = field(default_factory=dict)
    labeled_frames: set = field(default_factory=set)
    
    # Channels
    channel_manager: Optional[ChannelManager] = None
    active_channel_id: Optional[str] = None
    
    # View state
    zoom_factor: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    
    # Tool state (FIXED - use default_factory for enum)
    current_tool: ToolType = field(default_factory=lambda: ToolType.DRAW)
    brush_size: float = Config.BRUSH_SIZE_DEFAULT
    
    # Opacity
    mask_opacity: float = Config.CHANNEL_OPACITY
    
    # Enhancement settings
    brightness: float = 1.0
    contrast: float = 1.0
    gaussian_blur: int = 0
    low_pass_filter: int = 0
    high_pass_filter: int = 0
    temporal_smoothing: int = 0
    
    # Metadata (FIXED - removed last_modified, use last_saved only)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_saved: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def update_last_saved(self):
        """Update last saved timestamp."""
        self.last_saved = datetime.datetime.now().isoformat()
    
    def __setstate__(self, state):
        """CRITICAL FIX: Handle unpickling with backward compatibility migration."""
        # Migrate old 'last_modified' → 'last_saved'
        if 'last_modified' in state and 'last_saved' not in state:
            state['last_saved'] = state['last_modified']
            log("Migrated session: last_modified → last_saved")
        
        # Remove old field to avoid conflicts
        state.pop('last_modified', None)
        
        # Apply state
        self.__dict__.update(state)

@dataclass
class UndoState:
    """State snapshot for undo functionality."""
    frame_index: int
    channel_id: str
    mask_before: np.ndarray
    mask_after: np.ndarray
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    operation_type: str = "draw"


class UndoStack:
    """Circular buffer for undo operations."""
    
    def __init__(self, max_size: int = Config.UNDO_STACK_SIZE):
        self.max_size = max_size
        self._stack: List[UndoState] = []
        self._position = -1
    
    def push(self, state: UndoState) -> None:
        """Push new undo state."""
        if self._position < len(self._stack) - 1:
            self._stack = self._stack[:self._position + 1]
        
        self._stack.append(state)
        
        if len(self._stack) > self.max_size:
            self._stack.pop(0)
        else:
            self._position += 1
    
    def can_undo(self) -> bool:
        return self._position >= 0 and len(self._stack) > 0
    
    def can_redo(self) -> bool:
        return self._position < len(self._stack) - 1
    
    def undo(self) -> Optional[UndoState]:
        if not self.can_undo():
            return None
        state = self._stack[self._position]
        self._position -= 1
        return state
    
    def redo(self) -> Optional[UndoState]:
        if not self.can_redo():
            return None
        self._position += 1
        return self._stack[self._position]
    
    def clear(self) -> None:
        self._stack.clear()
        self._position = -1
        gc.collect()


# =============================================================================
# IMAGE CACHING SYSTEM
# =============================================================================

class LRUCache:
    """
    Enhanced Least Recently Used (LRU) cache with sliding window pre-loading.
    
    Features:
    - LRU eviction for memory management
    - Sliding window pre-loading for smooth navigation
    - Background thread for async frame loading
    - Memory-aware auto-eviction
    """
    
    def __init__(self, max_size: int = Config.MAX_FRAMES_IN_MEMORY):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        
        # Pre-loading state
        self.preload_enabled = Config.AGGRESSIVE_CACHING
        self.preload_thread = None
        self.preload_stop_flag = threading.Event()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache with LRU promotion."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self._hits += 1
                PERFORMANCE_STATS['cache_hits'] += 1
                return value
            self._misses += 1
            PERFORMANCE_STATS['cache_misses'] += 1
            return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache with LRU eviction."""
        with self._lock:
            if key in self.cache:
                self.cache.pop(key)
            self.cache[key] = value
            
            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # Remove oldest
    
    def put_batch(self, items: List[Tuple[Any, Any]]) -> None:
        """Put multiple items efficiently."""
        with self._lock:
            for key, value in items:
                if key in self.cache:
                    self.cache.pop(key)
                self.cache[key] = value
            
            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def contains(self, key: Any) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self.cache
    
    def preload_window(self, current_idx: int, total_frames: int, 
                   load_func: callable) -> None:
        """
        Pre-load frames in sliding window around current frame - FIXED.
        
        Args:
            current_idx: Current frame index
            total_frames: Total number of frames
            load_func: Function to load frame (frame_idx) -> frame_data
        """
        if not self.preload_enabled:
            return
        
        # Calculate window
        window_size = Config.PRELOAD_WINDOW_SIZE
        start_idx = max(0, current_idx - window_size // 3)  # Small lookback
        end_idx = min(total_frames, current_idx + window_size)  # Larger lookahead
        
        # FIXED: Don't try to join from the same thread
        if self.preload_thread and self.preload_thread.is_alive():
            self.preload_stop_flag.set()
            try:
                self.preload_thread.join(timeout=0.1)  # ADD THIS
            except:
                pass
        
        # Start new preload thread
        self.preload_stop_flag.clear()
        self.preload_thread = threading.Thread(
            target=self._preload_worker,
            args=(start_idx, end_idx, current_idx, load_func),
            daemon=True,
            name=f"PreloadWorker-{current_idx}"
        )
        self.preload_thread.start()
        

    def _preload_worker(self, start_idx: int, end_idx: int, 
                   current_idx: int, load_func: callable) -> None:
        """Background worker for pre-loading frames - FIXED: No self-join."""
        # Prioritize frames closest to current
        indices = list(range(start_idx, end_idx))
        
        # Sort by distance from current (closest first)
        indices.sort(key=lambda x: abs(x - current_idx))
        
        for idx in indices:
            # Check stop flag
            if self.preload_stop_flag.is_set():
                debug_log("Preload worker stopped by flag")
                break
            
            # Skip if already cached
            cache_key = f"frame_{idx}_display"
            if self.contains(cache_key):
                continue
            
            try:
                # Load frame
                frame = load_func(idx, for_display=True)
                if frame is not None:
                    self.put(cache_key, frame)
                    debug_log(f"Preloaded frame {idx}")
            except Exception as e:
                # Silently ignore errors in background loading
                debug_log(f"Preload error for frame {idx}: {e}")
        
        debug_log(f"Preload worker finished ({start_idx}-{end_idx})")

    
    def clear(self) -> None:
        """Clear cache and stop preloading - FIXED."""
        # Stop preloading thread
        if self.preload_thread and self.preload_thread.is_alive():
            self.preload_stop_flag.set()
            
            # FIXED: Only join if we're NOT in the preload thread itself
            if threading.current_thread() != self.preload_thread:
                self.preload_thread.join(timeout=0.5)
        
        with self._lock:
            self.cache.clear()
            gc.collect()


    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'preload_active': self.preload_thread.is_alive() if self.preload_thread else False
            }
# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Monitor application performance and resource usage."""
    
    def __init__(self):
        self.process = None
        if PSUTIL_AVAILABLE:
            try:
                import psutil
                self.process = psutil.Process(os.getpid())
            except:
                pass
    
    def get_memory_usage_mb(self) -> float:
        if self.process:
            try:
                return self.process.memory_info().rss / (1024 * 1024)
            except:
                pass
        return 0.0


class MemoryManager:
    """Automatic memory management system."""
    
    def __init__(self, threshold_mb: int = Config.MEMORY_THRESHOLD_MB):
        self.threshold_mb = threshold_mb
        self.monitor = PerformanceMonitor()
        self.cleanup_callbacks: List[callable] = []
    
    def register_cleanup_callback(self, callback: callable) -> None:
        self.cleanup_callbacks.append(callback)
    
    def check_memory(self) -> bool:
        current_mb = self.monitor.get_memory_usage_mb()
        return current_mb > self.threshold_mb
    
    def cleanup(self) -> float:
        memory_before = self.monitor.get_memory_usage_mb()
        
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                debug_log(f"Cleanup callback error: {e}")
        
        gc.collect()
        memory_after = self.monitor.get_memory_usage_mb()
        freed = memory_before - memory_after
        
        PERFORMANCE_STATS['memory_cleanups'] += 1
        
        if freed > 0:
            log(f"Memory cleanup: freed {freed:.1f} MB")
        
        return freed
    
    def auto_cleanup_if_needed(self) -> bool:
        if self.check_memory():
            self.cleanup()
            return True
        return False


class PlaybackController:
    """
    Controls automatic frame playback with adjustable FPS.
    
    Features:
    - Play/pause with smooth frame transitions
    - Adjustable FPS (1-60)
    - Loop or stop at end
    - Background frame pre-loading during playback
    """
    
    def __init__(self, app):
        self.app = app
        self.playing = False
        self.fps = Config.DEFAULT_PLAYBACK_FPS
        self.loop = False
        self.playback_timer = None
        self._stop_flag = False
    
    def play(self) -> bool:
        """Start playback."""
        if not self.app.has_frames():
            return False
        
        if self.playing:
            return False
        
        self.playing = True
        self._stop_flag = False
        self._play_next_frame()
        log(f"Playback started at {self.fps} FPS")
        return True
    
    def pause(self) -> None:
        """Pause playback."""
        self.playing = False
        self._stop_flag = True
        if self.playback_timer:
            self.app.root.after_cancel(self.playback_timer)
            self.playback_timer = None
        log("Playback paused")
    
    def stop(self) -> None:
        """Stop playback and reset."""
        self.pause()
        log("Playback stopped")
    
    def toggle(self) -> bool:
        """Toggle play/pause."""
        if self.playing:
            self.pause()
            return False
        else:
            return self.play()
    
    def set_fps(self, fps: int) -> None:
        """Set playback FPS - FIXED: Restart playback with new speed."""
        old_fps = self.fps
        self.fps = max(Config.MIN_PLAYBACK_FPS, 
                    min(Config.MAX_PLAYBACK_FPS, fps))
        
        # If playing, restart with new FPS
        if self.playing and old_fps != self.fps:
            # Cancel current timer
            if self.playback_timer:
                self.app.root.after_cancel(self.playback_timer)
                self.playback_timer = None
            # Restart with new delay
            self._play_next_frame()
            log(f"Playback FPS changed to {self.fps} (restarted)")
        else:
            log(f"Playback FPS set to {self.fps}")
    
    def set_loop(self, loop: bool) -> None:
        """Enable/disable looping."""
        self.loop = loop
    
    def _play_next_frame(self) -> None:
        """Play next frame (internal)."""
        if not self.playing or self._stop_flag:
            return
        
        # Check if at end
        if self.app.current_frame_idx >= self.app.image_handler.metadata.total_frames - 1:
            if self.loop:
                self.app.set_current_frame(0)
            else:
                self.stop()
                return
        else:
            self.app.next_frame()
        
        # Schedule next frame
        delay_ms = int(1000 / self.fps)
        self.playback_timer = self.app.root.after(delay_ms, self._play_next_frame)
    
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self.playing

# =============================================================================
# TIF HANDLER
# =============================================================================

class TIFHandler:
    """Efficient handler for multi-frame TIF files."""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.metadata: Optional[ImageMetadata] = None
        self.frame_cache = LRUCache(max_size=Config.MAX_FRAMES_IN_MEMORY)
        self.perf_monitor = PerformanceMonitor()
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix.lower() not in Config.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {self.file_path.suffix}")
        
        self._analyze_file()
    
    def _analyze_file(self) -> None:
        """Analyze TIF file and extract metadata."""
        log(f"Analyzing TIF file: {self.file_path.name}")
        
        try:
            with Image.open(self.file_path) as img:
                original_width = img.width
                original_height = img.height
                
                test_frame = np.array(img)
                dtype_str = str(test_frame.dtype)
                
                max_size = Config.MAX_DISPLAY_SIZE
                if original_width > max_size or original_height > max_size:
                    scale = min(max_size / original_width, max_size / original_height)
                    display_width = int(original_width * scale)
                    display_height = int(original_height * scale)
                    scale_factor = scale
                else:
                    display_width = original_width
                    display_height = original_height
                    scale_factor = 1.0
                
                frame_count = 0
                try:
                    while True:
                        img.seek(frame_count)
                        frame_count += 1
                except EOFError:
                    pass
                
                file_size_mb = self.file_path.stat().st_size / (1024 * 1024)
                
                self.metadata = ImageMetadata(
                    file_path=self.file_path,
                    total_frames=frame_count,
                    original_width=original_width,
                    original_height=original_height,
                    display_width=display_width,
                    display_height=display_height,
                    scale_factor=scale_factor,
                    dtype=dtype_str,
                    frame_indexing=FrameIndexing.ZERO_BASED,
                    file_size_mb=file_size_mb
                )
                
                log(f"TIF analyzed: {frame_count} frames, "
                    f"{original_width}×{original_height} → "
                    f"{display_width}×{display_height}")
                
        except Exception as e:
            raise ValueError(f"Failed to analyze TIF: {e}")
    
    def load_frame(self, frame_idx: int, for_display: bool = True) -> Optional[np.ndarray]:
        """Load a specific frame from TIF file with intelligent caching."""
        if self.metadata is None:
            return None
        
        if frame_idx < 0 or frame_idx >= self.metadata.total_frames:
            return None
        
        cache_key = f"frame_{frame_idx}_{'display' if for_display else 'full'}"
        cached = self.frame_cache.get(cache_key)
        if cached is not None:
            # TRIGGER PRE-LOADING for smooth navigation
            if for_display and Config.AGGRESSIVE_CACHING:
                self.frame_cache.preload_window(
                    frame_idx, 
                    self.metadata.total_frames,
                    self.load_frame
                )
            return cached
        
        try:
            with Image.open(self.file_path) as img:
                img.seek(frame_idx)
                frame = np.array(img)
                
                frame = self._convert_dtype(frame)
                frame = self._convert_color_space(frame)
                
                if for_display and self.metadata.scale_factor != 1.0:
                    frame = cv2.resize(
                        frame,
                        (self.metadata.display_width, self.metadata.display_height),
                        interpolation=Config.RESIZE_METHOD
                    )
                
                self.frame_cache.put(cache_key, frame)
                PERFORMANCE_STATS['frame_loads'] += 1
                
                # TRIGGER PRE-LOADING for next frames
                if for_display and Config.AGGRESSIVE_CACHING:
                    self.frame_cache.preload_window(
                        frame_idx,
                        self.metadata.total_frames,
                        self.load_frame
                    )
                
                return frame
                
        except Exception as e:
            log(f"Error loading frame {frame_idx}: {e}", 'ERROR')
            return None

    def _convert_dtype(self, frame: np.ndarray) -> np.ndarray:
        """Convert various data types to uint8 for display."""
        if frame.dtype == np.uint8:
            return frame
        elif frame.dtype == np.uint16:
            return (frame >> 8).astype(np.uint8)
        elif frame.dtype in [np.float32, np.float64]:
            frame_min = frame.min()
            frame_max = frame.max()
            if frame_max > frame_min:
                frame = (frame - frame_min) / (frame_max - frame_min) * 255
            else:
                frame = frame * 255
            return np.clip(frame, 0, 255).astype(np.uint8)
        else:
            return frame.astype(np.uint8)
    
    def _convert_color_space(self, frame: np.ndarray) -> np.ndarray:
        """Convert various color spaces to RGB."""
        if len(frame.shape) == 2:
            return np.stack([frame, frame, frame], axis=-1)
        elif len(frame.shape) == 3:
            channels = frame.shape[2]
            if channels == 1:
                return np.repeat(frame, 3, axis=2)
            elif channels == 3:
                return frame
            elif channels == 4:
                return frame[:, :, :3]
        return frame
    
    def create_empty_mask(self, for_display: bool = True) -> np.ndarray:
        """Create empty mask matching image dimensions."""
        if self.metadata is None:
            return np.zeros((100, 100), dtype=np.uint8)
        
        if for_display:
            return np.zeros((self.metadata.display_height, self.metadata.display_width),
                          dtype=np.uint8)
        else:
            return np.zeros((self.metadata.original_height, self.metadata.original_width),
                          dtype=np.uint8)
    
    def upscale_mask(self, mask: np.ndarray) -> np.ndarray:
        """Upscale display-resolution mask to full resolution."""
        if self.metadata is None:
            return mask
        
        if mask.shape[:2] == (self.metadata.original_height, self.metadata.original_width):
            return mask
        
        return cv2.resize(
            mask,
            (self.metadata.original_width, self.metadata.original_height),
            interpolation=cv2.INTER_NEAREST
        )
    
    def clear_cache(self) -> None:
        self.frame_cache.clear()


# =============================================================================
# IMAGE ENHANCEMENT
# =============================================================================

class ImageEnhancer:
    """Real-time image enhancement with multiple filters."""
    
    def __init__(self):
        self.brightness = 0
        self.contrast = 1.0
        self.gaussian_blur = 0
        self.low_pass_filter = 0
        self.high_pass_filter = 0
        self.temporal_smoothing = False
        self.prev_filtered_frame = None
        self.kalman_alpha = 0.7
    
    def apply_filters(self, frame: np.ndarray) -> np.ndarray:
        """Apply all active filters to frame."""
        if frame is None:
            return None
        
        filtered = frame.astype(np.float32)
        
        if self.brightness != 0 or self.contrast != 1.0:
            filtered = filtered * self.contrast + self.brightness
            filtered = np.clip(filtered, 0, 255)
        
        if self.gaussian_blur > 0:
            ksize = self.gaussian_blur
            if ksize % 2 == 0:
                ksize += 1
            filtered = cv2.GaussianBlur(filtered, (ksize, ksize), 0)
        
        if self.low_pass_filter > 0:
            filtered = self._apply_low_pass(filtered, self.low_pass_filter)
        
        if self.high_pass_filter > 0:
            filtered = self._apply_high_pass(filtered, self.high_pass_filter)
        
        if self.temporal_smoothing:
            filtered = self._apply_temporal_smoothing(filtered)
        
        return filtered.astype(np.uint8)
    
    def _apply_low_pass(self, frame: np.ndarray, strength: int) -> np.ndarray:
        if strength <= 0:
            return frame
        kernel_size = int(1 + (strength / 10)) * 2 + 1
        kernel_size = max(3, min(kernel_size, 31))
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    def _apply_high_pass(self, frame: np.ndarray, strength: int) -> np.ndarray:
        if strength <= 0:
            return frame
        low_pass = self._apply_low_pass(frame, strength)
        weight = strength / 100.0
        high_pass = cv2.addWeighted(frame, 1 + weight, low_pass, -weight, 0)
        return np.clip(high_pass, 0, 255)
    
    def _apply_temporal_smoothing(self, frame: np.ndarray) -> np.ndarray:
        if self.prev_filtered_frame is None:
            self.prev_filtered_frame = frame.copy()
            return frame
        filtered = (self.kalman_alpha * self.prev_filtered_frame +
                   (1 - self.kalman_alpha) * frame)
        self.prev_filtered_frame = filtered.copy()
        return filtered
    
    def reset_temporal_smoothing(self) -> None:
        self.prev_filtered_frame = None


# =============================================================================
# OPTICAL FLOW ANALYSIS
# =============================================================================

class OpticalFlowAnalyzer:
    """Optical flow computation for motion analysis."""
    
    def __init__(self, method: str = Config.OPTICAL_FLOW_METHOD):
        self.method = method
        self.has_dual_tvl1 = False
        try:
            cv2.optflow.DualTVL1OpticalFlow_create()
            self.has_dual_tvl1 = True
        except:
            if method == 'dual_tv_l1':
                log("DualTVL1 not available, falling back to Farneback", 'WARNING')
                self.method = 'farneback'
    
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute optical flow between two frames."""
        if self.method == 'dual_tv_l1' and self.has_dual_tvl1:
            return self._compute_dual_tvl1(frame1, frame2)
        else:
            return self._compute_farneback(frame1, frame2)
    
    def _compute_dual_tvl1(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = optical_flow.calc(frame1, frame2, None)
        return flow
    
    def _compute_farneback(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None,
            pyr_scale=Config.FLOW_PYR_SCALE,
            levels=Config.FLOW_LEVELS,
            winsize=Config.FLOW_WINSIZE,
            iterations=Config.FLOW_ITERATIONS,
            poly_n=Config.FLOW_POLY_N,
            poly_sigma=Config.FLOW_POLY_SIGMA,
            flags=0
        )
        return flow
    
    def analyze_flow(self, flow: np.ndarray) -> Dict[str, float]:
        """Analyze optical flow and extract statistics."""
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        return {
            'mean_mag': float(np.mean(flow_mag)),
            'max_mag': float(np.max(flow_mag)),
            'p95_mag': float(np.percentile(flow_mag, 95)),
            'large_motion_ratio': float(np.mean(flow_mag > 10.0)),
        }


# =============================================================================
# PROPAGATION ENGINE
# =============================================================================

class PropagationEngine:
    """Optical flow-based mask propagation."""
    
    def __init__(self, image_handler: TIFHandler):
        self.image_handler = image_handler
        self.flow_analyzer = OpticalFlowAnalyzer()
    
    def propagate_forward(self, source_frame_idx: int,
                         source_annotation: FrameAnnotation,
                         channel_manager: ChannelManager) -> Tuple[FrameAnnotation, float]:
        """Propagate masks from source to next frame."""
        target_frame_idx = source_frame_idx + 1
        
        source_frame = self._load_gray(source_frame_idx)
        target_frame = self._load_gray(target_frame_idx)
        
        flow = self.flow_analyzer.compute_flow(source_frame, target_frame)
        confidence = self._calculate_confidence(flow)
        
        target_annotation = FrameAnnotation(
            frame_index=target_frame_idx,
            propagated_from=source_frame_idx
        )
        
        for channel in channel_manager.get_all_channels():
            source_mask = source_annotation.get_mask(channel.id)
            
            if source_mask is not None and np.any(source_mask):
                warped_mask = self._warp_mask(source_mask, flow)
                target_annotation.set_mask(channel.id, warped_mask)
        
        return target_annotation, confidence
    
    def propagate_backward(self, source_frame_idx: int,
                          source_annotation: FrameAnnotation,
                          channel_manager: ChannelManager) -> Tuple[FrameAnnotation, float]:
        """Propagate masks backward (to previous frame)."""
        target_frame_idx = source_frame_idx - 1
        
        source_frame = self._load_gray(source_frame_idx)
        target_frame = self._load_gray(target_frame_idx)
        
        flow = self.flow_analyzer.compute_flow(target_frame, source_frame)
        confidence = self._calculate_confidence(flow)
        
        target_annotation = FrameAnnotation(
            frame_index=target_frame_idx,
            propagated_from=source_frame_idx
        )
        
        for channel in channel_manager.get_all_channels():
            source_mask = source_annotation.get_mask(channel.id)
            
            if source_mask is not None and np.any(source_mask):
                warped_mask = self._warp_mask(source_mask, flow)
                target_annotation.set_mask(channel.id, warped_mask)
        
        return target_annotation, confidence
    
    def _load_gray(self, frame_idx: int) -> np.ndarray:
        """Load frame as grayscale."""
        frame = self.image_handler.load_frame(frame_idx, for_display=True)
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame
    
    def _warp_mask(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp mask using optical flow field.
        
        Flow format: flow[y, x] = [dx, dy]
        For each pixel at (x, y), it moves to (x + dx, y + dy)
        """
        h, w = mask.shape
        
        # Create coordinate grids for the target (warped) image
        # These represent where each pixel in the SOURCE should GO in the target
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Apply flow to get new coordinates
        # flow[..., 0] is horizontal displacement (dx)
        # flow[..., 1] is vertical displacement (dy)
        x_new = x_coords + flow[..., 0]
        y_new = y_coords + flow[..., 1]
        
        # Warp the mask using the flow field
        # cv2.remap reads from the source mask at the given coordinates
        warped = cv2.remap(
            mask,
            x_new,
            y_new,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped
    
    def _calculate_confidence(self, flow: np.ndarray) -> float:
        """Calculate propagation confidence based on flow field (0-100)."""
        stats = self.flow_analyzer.analyze_flow(flow)
        
        mean_mag = stats['mean_mag']
        large_motion_ratio = stats['large_motion_ratio']
        
        if mean_mag < 2.0:
            confidence = 95.0
        elif mean_mag < 5.0:
            confidence = 80.0
        elif mean_mag < 10.0:
            confidence = 60.0
        else:
            confidence = 40.0
        
        confidence -= large_motion_ratio * 30
        
        return max(0.0, min(100.0, confidence))


# =============================================================================
# RECENT SESSIONS MANAGER
# =============================================================================

class RecentSessionsManager:
    """Manages recent sessions list with persistence."""
    
    def __init__(self, max_sessions: int = Config.MAX_RECENT_SESSIONS):
        self.max_sessions = max_sessions
        self.recent_file = Path(Config.RECENT_SESSIONS_FILE)
        self.sessions: List[Dict] = []
        self._lock = threading.Lock()
        self.load()
    
    def add_session(self, session_dir: Path, session_state: SessionState) -> None:
        """Add or update session in recent list."""
        with self._lock:
            abs_path = str(session_dir.absolute())
            
            try:
                rel_path = str(session_dir.relative_to(Path.cwd()))
            except ValueError:
                rel_path = None
            
            entry = {
                "path_absolute": abs_path,
                "path_relative": rel_path,
                "name": session_dir.name,
                "last_opened": datetime.datetime.now().isoformat(),
                "frame_count": session_state.image_metadata.total_frames,
                "labeled_frames": len(session_state.labeled_frames),
                "channels": [ch.name for ch in session_state.channel_manager.get_all_channels()],
                "tif_filename": session_state.image_metadata.file_path.name,
                "created_at": session_state.created_at if isinstance(session_state.created_at, str) else session_state.created_at.isoformat()
            }
            
            self.sessions = [s for s in self.sessions if s["path_absolute"] != abs_path]
            self.sessions.insert(0, entry)
            self.sessions = self.sessions[:self.max_sessions]
            
            self.save()
    
    def get_session_path(self, session_entry: Dict) -> Optional[Path]:
        """Get valid path for session (tries relative first, then absolute)."""
        if session_entry.get("path_relative"):
            rel_path = Path(session_entry["path_relative"])
            if rel_path.exists():
                return rel_path
        
        abs_path = Path(session_entry["path_absolute"])
        if abs_path.exists():
            return abs_path
        
        return None
    
    def load(self) -> bool:
        """Load recent sessions from JSON file."""
        try:
            if not self.recent_file.exists():
                self.sessions = []
                return True
            
            with open(self.recent_file, 'r') as f:
                data = json.load(f)
            
            self.sessions = data.get("sessions", [])
            log(f"Loaded {len(self.sessions)} recent sessions")
            return True
            
        except Exception as e:
            log(f"Error loading recent sessions: {e}", 'WARNING')
            self.sessions = []
            return False
    
    def save(self) -> bool:
        """Save recent sessions to JSON file."""
        try:
            data = {
                "version": VERSION,
                "last_updated": datetime.datetime.now().isoformat(),
                "max_sessions": self.max_sessions,
                "sessions": self.sessions
            }
            
            with open(self.recent_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            log(f"Error saving recent sessions: {e}", 'ERROR')
            return False
    
    def get_recent_sessions(self, count: int = 5) -> List[Dict]:
        """Get N most recent sessions (validates paths exist)."""
        valid_sessions = []
        
        for session in self.sessions[:count]:
            path = self.get_session_path(session)
            if path:
                valid_sessions.append(session)
        
        return valid_sessions
    
    def clear(self) -> None:
        """Clear all recent sessions."""
        with self._lock:
            self.sessions = []
            self.save()


# =============================================================================
# UI COMPONENTS - PROGRESS DIALOG
# =============================================================================

class ProgressDialog:
    """Animated progress dialog for long operations."""
    
    def __init__(self, parent, title: str = "Processing..."):
        self.parent = parent
        self.dialog = None
        
        try:
            self.dialog = tk.Toplevel(parent)
            self.dialog.title(title)
            self.dialog.configure(bg=Config.COLOR_BG_PRIMARY)
            self.dialog.transient(parent)
            self.dialog.grab_set()
            self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)
            
            main_frame = ttk.Frame(self.dialog, padding=20)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            self.status_label = ttk.Label(
                main_frame,
                text="Processing...",
                font=("Arial", 11)
            )
            self.status_label.pack(pady=(0, 10))
            
            self.progress_bar = ttk.Progressbar(
                main_frame,
                orient="horizontal",
                length=350,
                mode="determinate"
            )
            self.progress_bar.pack(pady=5, fill=tk.X)
            
            self.detail_label = ttk.Label(
                main_frame,
                text="",
                font=("Arial", 9)
            )
            self.detail_label.pack(pady=(5, 0))
            
            self._center_dialog()
            
        except Exception as e:
            log(f"Error creating progress dialog: {e}", 'ERROR')
    
    def _center_dialog(self):
        """Center dialog on parent window."""
        try:
            if self.dialog:
                self.dialog.update_idletasks()
                w = self.dialog.winfo_width()
                h = self.dialog.winfo_height()
                x = self.parent.winfo_rootx() + (self.parent.winfo_width() - w) // 2
                y = self.parent.winfo_rooty() + (self.parent.winfo_height() - h) // 2
                self.dialog.geometry(f"{w}x{h}+{x}+{y}")
        except Exception as e:
            debug_log(f"Error centering dialog: {e}")
    
    def update_progress(self, value: int, maximum: int, status: str = "", detail: str = ""):
        """Update progress bar and labels."""
        try:
            if not self.dialog:
                return
            
            self.progress_bar["maximum"] = maximum
            self.progress_bar["value"] = value
            
            if status:
                percentage = int((value / maximum) * 100) if maximum > 0 else 0
                self.status_label.config(text=f"{status} ({percentage}%)")
            
            if detail:
                self.detail_label.config(text=detail)
            
            self.dialog.update_idletasks()
            
        except Exception as e:
            debug_log(f"Error updating progress: {e}")
    
    def close(self):
        """Close the dialog."""
        try:
            if self.dialog:
                self.dialog.grab_release()
                self.dialog.destroy()
                self.dialog = None
        except Exception as e:
            debug_log(f"Error closing dialog: {e}")


# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    """
    Manages session persistence with dual-format storage.
    
    FIXED: This class now ONLY handles file I/O operations.
    Application state management is handled by MATApplication.
    """
    
    def __init__(self, project_dir: Path):
        """Initialize SessionManager with project directory."""
        self.project_dir = Path(project_dir)
    
    def save_session_to_file(self, session_state: SessionState) -> bool:
        """
        Save session state to file.
        
        Args:
            session_state: The SessionState object to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_file = self.project_dir / Config.SESSION_LOG_BINARY
            csv_file = self.project_dir / Config.SESSION_LOG_CSV
            json_file = self.project_dir / Config.SESSION_METADATA
            
            # Update timestamp
            session_state.update_last_saved()
            
            # Save binary (with masks)
            self._save_binary(session_state, session_file)
            
            # Save CSV (frame list)
            self._save_csv(session_state, csv_file)
            
            # Save JSON (metadata)
            self._save_json(session_state, json_file)
            
            log(f"Session saved to: {self.project_dir}")
            return True
            
        except Exception as e:
            log(f"Error saving session to file: {e}", 'ERROR')
            import traceback
            traceback.print_exc()
            return False
    
    def load_session_from_file(self, session_dir: Path) -> Optional[SessionState]:
        """
        Load session state from file.
        
        Args:
            session_dir: Directory containing session.dat
            
        Returns:
            SessionState object if successful, None otherwise
        """
        try:
            session_file = session_dir / Config.SESSION_LOG_BINARY
            
            if not session_file.exists():
                log(f"Session file not found: {session_file}", 'ERROR')
                return None
            
            # Load with migration support
            try:
                with open(session_file, 'rb') as f:
                    data = pickle.load(f)
                    # Handle old format with wrapper dict
                    if isinstance(data, dict) and 'session_state' in data:
                        session_state = data['session_state']
                    else:
                        session_state = data
            except TypeError as e:
                if 'unexpected keyword argument' in str(e):
                    log("Detected old session format, attempting migration...", 'WARNING')
                    with open(session_file, 'rb') as f:
                        old_data = pickle.load(f)
                    
                    if hasattr(old_data, '__dict__'):
                        state_dict = old_data.__dict__.copy()
                        state_dict = self._migrate_session_state(state_dict)
                        session_state = SessionState(**state_dict)
                        log("Successfully migrated old session format")
                    else:
                        raise
                else:
                    raise
            
            log(f"Session loaded from: {session_dir}")
            return session_state
            
        except Exception as e:
            log(f"Error loading session from file: {e}", 'ERROR')
            import traceback
            traceback.print_exc()
            return None
    
    def _migrate_session_state(self, old_state_dict: dict) -> dict:
        """Migrate old SessionState format to new format."""
        if 'last_modified' in old_state_dict and 'last_saved' not in old_state_dict:
            old_state_dict['last_saved'] = old_state_dict.pop('last_modified')
            log("Migrated: last_modified → last_saved")
        
        if 'created_at' not in old_state_dict:
            old_state_dict['created_at'] = datetime.datetime.now().isoformat()
        
        if 'last_saved' not in old_state_dict:
            old_state_dict['last_saved'] = datetime.datetime.now().isoformat()
        
        return old_state_dict
    
    def _save_binary(self, session_state: SessionState, path: Path):
        """Save session with masks to binary file."""
        data = {
            'version': VERSION,
            'session_state': session_state,
            'save_time': datetime.datetime.now(),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = path.stat().st_size / (1024 * 1024)
        debug_log(f"Binary session saved: {file_size:.2f} MB")
    
    def _save_csv(self, session_state: SessionState, path: Path):
        """Save frame list to CSV for easy viewing."""
        import csv
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Labeled', 'TotalPixels', 'IsKeyframe', 'Timestamp', 'EditCount'])
            
            for frame_idx in range(session_state.image_metadata.total_frames):
                if frame_idx in session_state.annotations:
                    ann = session_state.annotations[frame_idx]
                    writer.writerow([
                        frame_idx,
                        'Yes',
                        ann.total_labeled_pixels,
                        'Yes' if ann.is_keyframe else 'No',
                        ann.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        ann.edit_count
                    ])
                else:
                    writer.writerow([frame_idx, 'No', 0, 'No', '', 0])
    
    def _save_json(self, session_state: SessionState, path: Path):
        """Save metadata to JSON."""
        data = {
            'version': VERSION,
            'project_name': session_state.project_name,
            'created_at': session_state.created_at,
            'last_saved': session_state.last_saved,
            'total_frames': session_state.image_metadata.total_frames,
            'labeled_frames': len(session_state.labeled_frames),
            'channels': [ch.to_dict() for ch in session_state.channel_manager.get_all_channels()],
            'tif_file': {
                'original_path': str(session_state.image_metadata.file_path),
                'copied_to_session': True,
                'session_tif_path': Config.ORIGINAL_TIF_NAME
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

# =============================================================================
# FILE MANAGER
# =============================================================================

class FileManager:
    """Manages file operations: load TIF, save masks, export stacks."""
    
    def __init__(self, app):
        self.app = app
        self.current_project_dir: Optional[Path] = None
    
    def create_multichannel_structure(self, output_dir: Path, filename: str,
                                    source_tif_path: Path) -> Optional[Dict[str, Path]]:
        """Create project folder and copy original TIF.
        
        NEW STRUCTURE:
        /output/
          └── ProjectName_20251218_123456/
              ├── session.dat          ← Session files IN project folder
              ├── session.csv
              ├── session_info.json
              ├── original.tif
              ├── Image/
              ├── Overlay/
              └── Masks/               ← NEW parent folder
                  ├── Mask_All/        ← Combined masks
                  ├── Mask_High_Confidence/
                  ├── Mask_Medium_Confidence/
                  ├── Mask_Low_Confidence/
                  └── Mask_Borderline_Cases/
        """
        try:
            timestamp = get_timestamp()
            base_name = Path(filename).stem
            
            project_name = f"{base_name}_{timestamp}"
            project_dir = output_dir / project_name
            
            image_dir = project_dir / Config.SUBFOLDER_IMAGES
            overlay_dir = project_dir / Config.SUBFOLDER_OVERLAYS
            masks_parent_dir = project_dir / Config.SUBFOLDER_MASKS
            
            ensure_dir(project_dir)
            ensure_dir(image_dir)
            ensure_dir(overlay_dir)
            ensure_dir(masks_parent_dir)
            
            # Copy original TIF to session folder
            tif_copy_path = project_dir / Config.ORIGINAL_TIF_NAME
            
            try:
                log(f"Copying TIF to session folder: {source_tif_path.name}")
                shutil.copy2(source_tif_path, tif_copy_path)
                log(f"TIF copied: {format_bytes(tif_copy_path.stat().st_size)}")
            except Exception as e:
                log(f"Warning: Could not copy TIF file: {e}", 'WARNING')
            
            # Create channel-specific folders INSIDE Masks/
            channel_manager = self.app.session_state.channel_manager
            mask_dirs = {}
            for channel in channel_manager.get_all_channels():
                mask_dir = masks_parent_dir / channel.folder_name
                ensure_dir(mask_dir)
                mask_dirs[channel.id] = mask_dir
            
            # Create Mask_All folder INSIDE Masks/
            mask_all_dir = masks_parent_dir / Config.SUBFOLDER_MASK_ALL
            ensure_dir(mask_all_dir)
            
            self.current_project_dir = project_dir
            log(f"Created multi-channel project structure: {project_name}")
            
            result = {
                'main': project_dir,
                'image': image_dir,
                'overlay': overlay_dir,
                'masks_parent': masks_parent_dir,
                'mask_all': mask_all_dir,
                'tif': tif_copy_path
            }
            result.update(mask_dirs)
            
            return result
            
        except Exception as e:
            log(f"Error creating project structure: {e}", 'ERROR')
            return None
    
    def get_frame_filename(self, base_name: str, frame_idx: int,
                          frame_indexing: FrameIndexing) -> str:
        """Generate standardized frame filename."""
        if frame_indexing == FrameIndexing.ONE_BASED:
            display_idx = frame_idx + 1
        else:
            display_idx = frame_idx
        
        padded = str(display_idx).zfill(Config.FRAME_DIGIT_PADDING)
        return f"{base_name}_frame_{padded}.tif"
    
    def save_current_frame(self, output_dir: Path) -> bool:
        """Save current frame with all channel masks."""
        try:
            if not self.app.has_frames():
                log("No frames to save", 'WARNING')
                return False
            
            image_handler = self.app.image_handler
            current_idx = self.app.current_frame_idx
            
            original_frame = image_handler.load_frame(current_idx, for_display=False)
            if original_frame is None:
                log("Failed to load frame", 'ERROR')
                return False
            
            annotation = self.app.get_current_annotation()
            if annotation is None or not annotation.has_annotations:
                log("No annotations to save", 'WARNING')
                return False
            
            #Detect if output directory has changed
            need_new_structure = False

            if self.current_project_dir is None:
                need_new_structure = True
            else:
                current_output_dir = self.current_project_dir.parent.resolve()
                new_output_dir = output_dir.resolve()

                if current_output_dir != new_output_dir:
                    log(f"Output directory changed: {current_output_dir} → {new_output_dir}")
                    log("Creating NEW project structure in new location...")
                    need_new_structure = True
            
            # Create new structure or use existing
            if need_new_structure:
                base_name = image_handler.metadata.file_path.stem
                folders = self.create_multichannel_structure(
                    output_dir, base_name, image_handler.metadata.file_path
                )
                if folders is None:
                    return False
            else:
                folders = self._get_existing_folders()
            
            base_name = image_handler.metadata.file_path.stem
            frame_indexing = image_handler.metadata.frame_indexing
            filename = self.get_frame_filename(base_name, current_idx, frame_indexing)
            
            # Save original image
            img_path = folders['image'] / filename
            pil_frame = Image.fromarray(original_frame)
            pil_frame.save(img_path, compression=Config.EXPORT_COMPRESSION)
            
            # Create combined mask (binary 255 for ANY annotation)
            h, w = image_handler.metadata.original_height, image_handler.metadata.original_width
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Save each channel mask
            channel_manager = self.app.session_state.channel_manager
            channel_stats = {}
            
            for channel in channel_manager.get_all_channels():
                display_mask = annotation.get_mask(channel.id)
                if display_mask is not None and np.any(display_mask):
                    full_mask = image_handler.upscale_mask(display_mask)
                    mask_path = folders[channel.id] / filename
                    pil_mask = Image.fromarray(full_mask)
                    pil_mask.save(mask_path, compression=Config.EXPORT_COMPRESSION)
                    channel_stats[channel.id] = np.sum(full_mask > 0)
                    
                    # Add to combined mask (binary 255)
                    combined_mask[full_mask > 0] = 255
            
            # Save combined mask to Mask_All
            if np.any(combined_mask):
                mask_all_path = folders['mask_all'] / filename
                pil_combined = Image.fromarray(combined_mask)
                pil_combined.save(mask_all_path, compression=Config.EXPORT_COMPRESSION)
            
            # Create composite overlay
            overlay_frame = original_frame.copy()
            if len(overlay_frame.shape) == 2:
                overlay_frame = np.stack([overlay_frame] * 3, axis=-1)
            
            for channel in reversed(channel_manager.get_all_channels()):
                display_mask = annotation.get_mask(channel.id)
                if display_mask is not None and np.any(display_mask):
                    full_mask = image_handler.upscale_mask(display_mask)
                    mask_pixels = np.where(full_mask > 0)
                    if len(mask_pixels[0]) > 0:
                        overlay_frame[mask_pixels] = channel.color_rgb
            
            overlay_path = folders['overlay'] / filename
            pil_overlay = Image.fromarray(overlay_frame)
            pil_overlay.save(overlay_path, compression=Config.EXPORT_COMPRESSION)
            
            # Auto-save session
            if Config.AUTOSAVE_ON_FRAME_SAVE and self.current_project_dir:
                self.app.project_dir = self.current_project_dir
                self.app.save_session()
            
            total_labeled = sum(channel_stats.values())
            log(f"Frame {current_idx} saved: {total_labeled} total pixels")
            
            messagebox.showinfo("Saved", f"Frame {current_idx + 1} saved successfully!")
            return True
            
        except Exception as e:
            log(f"Error saving current frame: {e}", 'ERROR')
            import traceback
            traceback.print_exc()
            return False
    
    def _get_existing_folders(self) -> Dict[str, Path]:
        """Get folder paths for existing project."""
        if self.current_project_dir is None:
            return {}
        
        masks_parent_dir = self.current_project_dir / Config.SUBFOLDER_MASKS
        
        result = {
            'main': self.current_project_dir,
            'image': self.current_project_dir / Config.SUBFOLDER_IMAGES,
            'overlay': self.current_project_dir / Config.SUBFOLDER_OVERLAYS,
            'masks_parent': masks_parent_dir,
            'mask_all': masks_parent_dir / Config.SUBFOLDER_MASK_ALL,
        }
        
        channel_manager = self.app.session_state.channel_manager
        for channel in channel_manager.get_all_channels():
            result[channel.id] = masks_parent_dir / channel.folder_name
        
        return result


# =============================================================================
# UI - OPTIMIZED CANVAS
# =============================================================================

class OptimizedCanvas(ttk.Frame):
    """High-performance drawing canvas with zoom, pan, and drawing."""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        self.drawing = False
        self.last_x, self.last_y = 0, 0
        self.last_draw_time = 0
        self.draw_buffer = []
        
        # NEW: Contour drawing state
        self.contour_points = []  # List of (x, y) points for current contour
        self.contour_canvas_items = []  # Canvas item IDs for visual feedback
        
        self.canvas = tk.Canvas(self, bg=Config.COLOR_BG_CANVAS, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.current_photo = None
        self._photo_refs = []  # FIXED: Keep references to prevent garbage collection
        
        self._setup_bindings()
    
    def _setup_bindings(self):
        """Setup all mouse and keyboard event bindings."""
        self.canvas.bind("<Button-1>", self._start_drawing)
        self.canvas.bind("<B1-Motion>", self._draw_continuous)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drawing)
        
        self.canvas.bind("<MouseWheel>", self._zoom_with_mousewheel)
        self.canvas.bind("<Button-4>", self._zoom_in_event)
        self.canvas.bind("<Button-5>", self._zoom_out_event)
        
        self.canvas.bind("<Button-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan_motion)
        self.canvas.bind("<ButtonRelease-2>", self._stop_pan)
        self.canvas.bind("<Button-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._pan_motion)
        self.canvas.bind("<ButtonRelease-3>", self._stop_pan)
    
    def _start_drawing(self, event):
        """Start drawing operation."""
        try:
            if not self.app.has_frames():
                return
            
            # FIXED: Check if channel was switched during drawing
            if self.drawing and self.app.active_channel_id != self._drawing_channel_id:
                log("Channel switched during drawing - restarting operation", 'WARNING')
                self.drawing = False
            
            # Store active channel at start of drawing
            self._drawing_channel_id = self.app.active_channel_id
            
            self.drawing = True
            self.last_x, self.last_y = event.x, event.y
            
            # FIXED: Save mask_before for undo
            active_channel_id = self.app.active_channel_id
            mask = self.app.get_current_channel_mask()
            if mask is not None:
                self._undo_mask_before = mask.copy()  # Store for later
            
            # Handle contour tool differently
            if self.app.current_tool == ToolType.CONTOUR:
                self._start_contour(event.x, event.y)
            else:
                self.draw_buffer = []
                self._draw_at_position(event.x, event.y)
            
            PERFORMANCE_STATS['total_draw_operations'] += 1
            
        except Exception as e:
            log(f"Error starting draw: {e}", 'ERROR')
    
    def _start_contour(self, canvas_x: int, canvas_y: int):
        """Start contour drawing."""
        # Clear previous contour
        self._clear_contour_visual()
        self.contour_points = []
        
        # Add first point
        img_x, img_y = self._canvas_to_image_coords(canvas_x, canvas_y)
        self.contour_points.append((img_x, img_y))
        
        # Draw point on canvas
        item_id = self.canvas.create_oval(
            canvas_x - 3, canvas_y - 3,
            canvas_x + 3, canvas_y + 3,
            fill='yellow', outline='red', width=2
        )
        self.contour_canvas_items.append(item_id)
    
    def _draw_continuous(self, event):
        """Handle continuous drawing with buffering."""
        if not self.drawing:
            return
        
        # FIXED: Check for channel switch during drawing
        if self.app.active_channel_id != self._drawing_channel_id:
            log("Channel switched - stopping draw operation", 'WARNING')
            self._stop_drawing(event)
            return
        
        # Handle contour tool
        if self.app.current_tool == ToolType.CONTOUR:
            self._add_contour_point(event.x, event.y)
            return
        
        current_time = time.time()
        
        if current_time - self.last_draw_time < Config.DRAW_THROTTLE_MS / 1000:
            self.draw_buffer.append((event.x, event.y))
            return
        
        if self.draw_buffer:
            for x, y in self.draw_buffer:
                self._draw_at_position(x, y)
            self.draw_buffer = []
        
        self._draw_at_position(event.x, event.y)
        self.last_draw_time = current_time
    
    def _add_contour_point(self, canvas_x: int, canvas_y: int):
        """Add point to contour."""
        if len(self.contour_points) == 0:
            return
        
        img_x, img_y = self._canvas_to_image_coords(canvas_x, canvas_y)
        
        # Check distance from last point (avoid duplicate points)
        last_x, last_y = self.contour_points[-1]
        dist = np.sqrt((img_x - last_x)**2 + (img_y - last_y)**2)
        
        if dist < 2:  # Minimum distance threshold
            return
        
        self.contour_points.append((img_x, img_y))
        
        # Draw line from last point
        last_canvas_x, last_canvas_y = self._image_to_canvas_coords(last_x, last_y)
        
        item_id = self.canvas.create_line(
            last_canvas_x, last_canvas_y,
            canvas_x, canvas_y,
            fill='yellow', width=2
        )
        self.contour_canvas_items.append(item_id)
        
        # Draw point
        item_id = self.canvas.create_oval(
            canvas_x - 3, canvas_y - 3,
            canvas_x + 3, canvas_y + 3,
            fill='yellow', outline='red', width=2
        )
        self.contour_canvas_items.append(item_id)
    
    def _draw_at_position(self, canvas_x: int, canvas_y: int):
        """Draw at specific canvas position (multi-channel aware)."""
        try:
            if not self.drawing or not self.app.has_frames():
                return
            
            # Skip for contour tool
            if self.app.current_tool == ToolType.CONTOUR:
                return
            
            img_x, img_y = self._canvas_to_image_coords(canvas_x, canvas_y)
            
            mask = self.app.get_current_channel_mask()
            if mask is None:
                return
            
            h, w = mask.shape
            if not (0 <= img_x < w and 0 <= img_y < h):
                return
            
            brush_radius = max(1, int(round(self.app.brush_size)))
            
            if self.app.current_tool == ToolType.DRAW:
                cv2.circle(mask, (img_x, img_y), brush_radius, 255, -1)
            elif self.app.current_tool == ToolType.ERASE:
                cv2.circle(mask, (img_x, img_y), brush_radius, 0, -1)
            elif self.app.current_tool == ToolType.ERASE_ADMIN:
                annotation = self.app.get_current_annotation()
                if annotation:
                    channel_manager = self.app.session_state.channel_manager
                    for channel in channel_manager.get_all_channels():
                        channel_mask = annotation.get_mask(channel.id)
                        if channel_mask is not None:
                            cv2.circle(channel_mask, (img_x, img_y), brush_radius, 0, -1)
                            annotation.set_mask(channel.id, channel_mask)
            
            if self.app.current_tool != ToolType.ERASE_ADMIN:
                annotation = self.app.get_current_annotation()
                if annotation:
                    annotation.set_mask(self.app.active_channel_id, mask)
            
        except Exception as e:
            debug_log(f"Error in draw_at_position: {e}")
    
    def _stop_drawing(self, event):
        """Stop drawing and update display."""
        if self.drawing:
            # Handle contour tool completion
            if self.app.current_tool == ToolType.CONTOUR:
                self._complete_contour()
            else:
                if self.draw_buffer:
                    for x, y in self.draw_buffer:
                        self._draw_at_position(x, y)
                    self.draw_buffer = []
                
                # FIXED: Now save mask_after for undo
                if hasattr(self, '_undo_mask_before'):
                    mask_after = self.app.get_current_channel_mask()
                    if mask_after is not None:
                        undo_state = UndoState(
                            frame_index=self.app.current_frame_idx,
                            channel_id=self.app.active_channel_id,
                            mask_before=self._undo_mask_before,
                            mask_after=mask_after.copy(),
                            operation_type=self.app.current_tool.value
                        )
                        self.app.undo_stack.push(undo_state)
                    delattr(self, '_undo_mask_before')
            
            # Mark frame as modified
            self.app.frame_modified[self.app.current_frame_idx] = True
            
            self.app.update_display_now()
        
        self.drawing = False
    
    def _complete_contour(self):
        """Complete contour based on selected mode."""
        if len(self.contour_points) < 3:
            messagebox.showwarning("Contour", "Need at least 3 points to create a contour", parent=self.app.root)
            self._clear_contour_visual()
            return
        
        # Close contour visually
        first_x, first_y = self.contour_points[0]
        last_x, last_y = self.contour_points[-1]
        
        first_canvas_x, first_canvas_y = self._image_to_canvas_coords(first_x, first_y)
        last_canvas_x, last_canvas_y = self._image_to_canvas_coords(last_x, last_y)
        
        item_id = self.canvas.create_line(
            last_canvas_x, last_canvas_y,
            first_canvas_x, first_canvas_y,
            fill='green', width=3
        )
        self.contour_canvas_items.append(item_id)
        
        # Check contour mode from radio buttons
        mode = self.app.contour_mode_var.get()
        
        if mode == "ask":
            # Show dialog
            self._show_contour_mode_dialog()
        elif mode == "fill":
            # Apply fill directly
            self._apply_contour_fill_direct()
        elif mode == "outline":
            # Apply outline directly
            self._apply_contour_outline_direct()

    def _show_contour_mode_dialog(self):
        """Show dialog asking user to choose fill or outline."""
        dialog = tk.Toplevel(self.app.root)
        dialog.title("Contour Mode")
        dialog.geometry("300x150")
        dialog.transient(self.app.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_reqwidth() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_reqheight() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            main_frame,
            text="How to draw this contour?",
            font=("Arial", 11, "bold")
        ).pack(pady=(0, 15))
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def fill_action():
            dialog.destroy()
            self._apply_contour_fill_direct()
        
        def outline_action():
            dialog.destroy()
            self._apply_contour_outline_direct()
        
        def cancel_action():
            self._clear_contour_visual()
            if hasattr(self, '_undo_mask_before'):
                delattr(self, '_undo_mask_before')
            dialog.destroy()
        
        ttk.Button(
            button_frame,
            text="Fill Contour",
            command=fill_action,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Outline Only",
            command=outline_action,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            main_frame,
            text="Cancel",
            command=cancel_action,
            width=10
        ).pack(pady=(10, 0))


    def _apply_contour_fill_direct(self):
        """Fill contour without dialog."""
        try:
            mask = self.app.get_current_channel_mask()
            if mask is None:
                self._clear_contour_visual()
                return
            
            if hasattr(self, '_undo_mask_before'):
                mask_before = self._undo_mask_before.copy()
            else:
                mask_before = mask.copy()
            
            contour_array = np.array(self.contour_points, dtype=np.int32)
            cv2.fillPoly(mask, [contour_array], 255)
            
            annotation = self.app.get_current_annotation()
            if annotation:
                annotation.set_mask(self.app.active_channel_id, mask)
            
            undo_state = UndoState(
                frame_index=self.app.current_frame_idx,
                channel_id=self.app.active_channel_id,
                mask_before=mask_before,
                mask_after=mask.copy(),
                operation_type="contour_fill"
            )
            self.app.undo_stack.push(undo_state)
            
            self.app.frame_modified[self.app.current_frame_idx] = True
            
            log(f"Filled contour with {len(self.contour_points)} points")
            
            self._clear_contour_visual()
            self.app.update_display_now()
            
        except Exception as e:
            log(f"Error filling contour: {e}", 'ERROR')
            self._clear_contour_visual()

    def _apply_contour_outline_direct(self):
        """Draw contour outline without dialog."""
        try:
            mask = self.app.get_current_channel_mask()
            if mask is None:
                self._clear_contour_visual()
                return
            
            if hasattr(self, '_undo_mask_before'):
                mask_before = self._undo_mask_before.copy()
            else:
                mask_before = mask.copy()
            
            contour_array = np.array(self.contour_points, dtype=np.int32)
            brush_radius = max(1, int(round(self.app.brush_size)))
            cv2.polylines(mask, [contour_array], isClosed=True, color=255, thickness=brush_radius)
            
            annotation = self.app.get_current_annotation()
            if annotation:
                annotation.set_mask(self.app.active_channel_id, mask)
            
            undo_state = UndoState(
                frame_index=self.app.current_frame_idx,
                channel_id=self.app.active_channel_id,
                mask_before=mask_before,
                mask_after=mask.copy(),
                operation_type="contour_outline"
            )
            self.app.undo_stack.push(undo_state)
            
            self.app.frame_modified[self.app.current_frame_idx] = True
            
            log(f"Drew contour outline with {len(self.contour_points)} points")
            
            self._clear_contour_visual()
            self.app.update_display_now()
            
        except Exception as e:
            log(f"Error drawing contour: {e}", 'ERROR')
            self._clear_contour_visual()


    def _cancel_contour(self, dialog):
        """Cancel contour without applying."""
        self._clear_contour_visual()
        if hasattr(self, '_undo_mask_before'):
            delattr(self, '_undo_mask_before')
        dialog.destroy()
    
    def _clear_contour_visual(self):
        """Clear contour visual feedback from canvas."""
        for item_id in self.contour_canvas_items:
            try:
                self.canvas.delete(item_id)
            except:
                pass
        self.contour_canvas_items = []
        self.contour_points = []
    
    def _canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> Tuple[int, int]:
        """Convert canvas coordinates to image coordinates."""
        try:
            # FIXED: Check if display_info is None
            display_info = self.app.display_info
            if not display_info:
                return 0, 0
            
            img_x, img_y, img_w, img_h = display_info
            
            if img_x <= canvas_x <= img_x + img_w and img_y <= canvas_y <= img_y + img_h:
                rel_x = (canvas_x - img_x) / img_w
                rel_y = (canvas_y - img_y) / img_h
                
                image_handler = self.app.image_handler
                if image_handler and image_handler.metadata:
                    display_w = image_handler.metadata.display_width
                    display_h = image_handler.metadata.display_height
                    
                    return int(rel_x * display_w), int(rel_y * display_h)
            
            return 0, 0
            
        except Exception as e:
            debug_log(f"Error converting coordinates: {e}")
            return 0, 0
    
    def _image_to_canvas_coords(self, img_x: int, img_y: int) -> Tuple[int, int]:
        """Convert image coordinates to canvas coordinates."""
        try:
            # FIXED: Check if display_info is None
            display_info = self.app.display_info
            if not display_info:
                return 0, 0
            
            canvas_img_x, canvas_img_y, img_w, img_h = display_info
            
            image_handler = self.app.image_handler
            if image_handler and image_handler.metadata:
                display_w = image_handler.metadata.display_width
                display_h = image_handler.metadata.display_height
                
                rel_x = img_x / display_w
                rel_y = img_y / display_h
                
                canvas_x = int(canvas_img_x + rel_x * img_w)
                canvas_y = int(canvas_img_y + rel_y * img_h)
                
                return canvas_x, canvas_y
            
            return 0, 0
            
        except Exception as e:
            debug_log(f"Error converting coordinates: {e}")
            return 0, 0
    
    def _start_pan(self, event):
        self.canvas.config(cursor="fleur")
        self.last_x, self.last_y = event.x, event.y
    
    def _pan_motion(self, event):
        try:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            self.pan_x += dx
            self.pan_y += dy
            self.last_x, self.last_y = event.x, event.y
            self.app.schedule_display_update()
        except Exception as e:
            debug_log(f"Error panning: {e}")
    
    def _stop_pan(self, event):
        self.canvas.config(cursor="")
    
    def _zoom_with_mousewheel(self, event):
        try:
            if hasattr(event, 'delta'):
                zoom_in = event.delta > 0
            else:
                zoom_in = True
            self._apply_zoom(zoom_in, event.x, event.y)
        except Exception as e:
            debug_log(f"Error zooming: {e}")
    
    def _zoom_in_event(self, event):
        self._apply_zoom(True, event.x, event.y)
    
    def _zoom_out_event(self, event):
        self._apply_zoom(False, event.x, event.y)
    
    def _apply_zoom(self, zoom_in: bool, mouse_x: int, mouse_y: int):
        """Apply zoom centered on mouse position."""
        try:
            if not self.app.has_frames():
                return
            
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            if canvas_w <= 1 or canvas_h <= 1:
                return
            
            factor = Config.ZOOM_STEP if zoom_in else (1.0 / Config.ZOOM_STEP)
            new_zoom = self.zoom_factor * factor
            new_zoom = max(Config.ZOOM_MIN, min(Config.ZOOM_MAX, new_zoom))
            
            if abs(new_zoom - self.zoom_factor) < 0.01:
                return
            
            image_handler = self.app.image_handler
            if not image_handler or not image_handler.metadata:
                return
            
            display_w = image_handler.metadata.display_width
            display_h = image_handler.metadata.display_height
            
            old_img_w = int(display_w * self.zoom_factor)
            old_img_h = int(display_h * self.zoom_factor)
            old_img_x = (canvas_w - old_img_w) // 2 + self.pan_x
            old_img_y = (canvas_h - old_img_h) // 2 + self.pan_y
            
            if old_img_w > 0 and old_img_h > 0:
                mouse_img_x = (mouse_x - old_img_x) / old_img_w
                mouse_img_y = (mouse_y - old_img_y) / old_img_h
                mouse_img_x = max(0, min(1, mouse_img_x))
                mouse_img_y = max(0, min(1, mouse_img_y))
            else:
                mouse_img_x = 0.5
                mouse_img_y = 0.5
            
            self.zoom_factor = new_zoom
            
            new_img_w = int(display_w * self.zoom_factor)
            new_img_h = int(display_h * self.zoom_factor)
            
            new_mouse_img_x = mouse_img_x * new_img_w
            new_mouse_img_y = mouse_img_y * new_img_h
            
            new_img_x = mouse_x - new_mouse_img_x
            new_img_y = mouse_y - new_mouse_img_y
            
            center_x = (canvas_w - new_img_w) // 2
            center_y = (canvas_h - new_img_h) // 2
            
            self.pan_x = new_img_x - center_x
            self.pan_y = new_img_y - center_y
            
            self.app.schedule_display_update()
            
        except Exception as e:
            log(f"Error applying zoom: {e}", 'ERROR')
    
    def reset_view(self):
        """Reset zoom and pan to default."""
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._clear_contour_visual()  # Clear any contours
        self.app.schedule_display_update()
    
    def clear_canvas(self):
        """Clear canvas contents."""
        try:
            self.canvas.delete("all")
            self.current_photo = None
            # FIXED: Clear photo references
            self._photo_refs.clear()
            self._clear_contour_visual()
        except Exception as e:
            debug_log(f"Error clearing canvas: {e}")


# =============================================================================
# UI - TOOLBAR
# =============================================================================

class ToolbarManager(ttk.Frame):
    """Manages toolbar with tool selection and brush size control."""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.tool_indicator = ttk.Label(
            self,
            text="DRAW",
            font=("Arial", 10, "bold"),
            foreground=Config.COLOR_ACCENT,
            width=12
        )
        self.tool_indicator.pack(side=tk.LEFT, padx=10)
        
        self.memory_label = ttk.Label(
            self,
            text="Memory: 0 MB",
            font=("Arial", 9)
        )
        self.memory_label.pack(side=tk.LEFT, padx=10)
        
        # Frame status indicator
        self.frame_status_indicator = ttk.Label(
            self,
            text="●",
            font=("Arial", 14, "bold"),
            foreground="#888888"
        )
        self.frame_status_indicator.pack(side=tk.LEFT, padx=5)
        
        self.frame_status_label = ttk.Label(
            self,
            text="No Frame",
            font=("Arial", 9)
        )
        self.frame_status_label.pack(side=tk.LEFT, padx=5)
        
        brush_frame = ttk.Frame(self)
        brush_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(brush_frame, text="Brush:", font=("Arial", 9)).pack(side=tk.LEFT)
        
        self.brush_scale = ttk.Scale(
            brush_frame,
            from_=Config.BRUSH_SIZE_MIN,
            to=Config.BRUSH_SIZE_MAX,
            orient=tk.HORIZONTAL,
            command=lambda x: self.app.update_brush_size(x),
            length=150
        )
        self.brush_scale.set(Config.BRUSH_SIZE_DEFAULT)
        self.brush_scale.pack(side=tk.LEFT, padx=5)
        
        self.brush_value_label = ttk.Label(
            brush_frame,
            text=f"{Config.BRUSH_SIZE_DEFAULT:.1f}",
            width=4
        )
        self.brush_value_label.pack(side=tk.LEFT)
    
    def update_tool_indicator(self, tool: ToolType):
        """Update tool indicator with color coding."""
        if tool == ToolType.DRAW:
            self.tool_indicator.config(text="DRAW", foreground=Config.COLOR_ACCENT)
        elif tool == ToolType.ERASE:
            self.tool_indicator.config(text="ERASE", foreground=Config.COLOR_WARNING)
        elif tool == ToolType.ERASE_ADMIN:
            self.tool_indicator.config(text="ERASE ALL", foreground=Config.COLOR_ERROR)
        elif tool == ToolType.CONTOUR:
            self.tool_indicator.config(text="CONTOUR", foreground="#00BFFF")
    
    def update_memory_display(self, memory_mb: float):
        """Update memory indicator with color coding."""
        # Get cache stats
        if hasattr(self.app, 'image_handler') and self.app.image_handler:
            cache = self.app.image_handler.frame_cache
            stats = cache.get_stats()
            cache_info = f" | Cache: {stats['size']}/{stats['max_size']} ({stats['hit_rate']:.0f}%)"
        else:
            cache_info = ""
        
        self.memory_label.config(text=f"Mem: {memory_mb:.0f}MB{cache_info}")
        
        if memory_mb > Config.MEMORY_THRESHOLD_MB:
            color = Config.COLOR_ERROR
        elif memory_mb > Config.MEMORY_THRESHOLD_MB * 0.8:
            color = Config.COLOR_WARNING
        else:
            color = Config.COLOR_ACCENT
        
        self.memory_label.config(foreground=color)
    
    def update_brush_value(self, value: float):
        """Update brush size value label."""
        self.brush_value_label.config(text=f"{value:.1f}")
    
    def update_frame_status(self, has_annotations: bool, is_saved: bool):
        """Update frame status indicator.
        
        Green: Frame has annotations AND is saved
        Orange: Frame has annotations but NOT saved
        Red: Frame has no annotations
        """
        if not has_annotations:
            # Red - not labeled
            self.frame_status_indicator.config(foreground="#FF0000")
            self.frame_status_label.config(text="Not Labeled")
        elif not is_saved:
            # Orange - labeled but not saved
            self.frame_status_indicator.config(foreground="#FFA500")
            self.frame_status_label.config(text="Modified")
        else:
            # Green - labeled and saved
            self.frame_status_indicator.config(foreground="#00FF00")
            self.frame_status_label.config(text="Saved")


# =============================================================================
# UI - NAVIGATION PANEL
# =============================================================================

class NavigationPanel(ttk.Frame):
    """Frame navigation with slider, skip control, and frame counter."""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.slider_updating = False
        self.slider_update_pending = None
        self.slider_update_delay_ms = 100
        
        # === LEFT SIDE: Frame controls ===
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, padx=5)
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(side=tk.TOP)
        
        ttk.Button(btn_frame, text="◀", command=self.app.prev_frame, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="▶", command=self.app.next_frame, width=3).pack(side=tk.LEFT, padx=1)
        
        self.frame_label = ttk.Label(left_frame, text="Frame: 0/0", font=("Arial", 10))
        self.frame_label.pack(side=tk.TOP, pady=(5, 0))
        
        # === MIDDLE: Playback controls ===
        playback_frame = ttk.LabelFrame(self, text="▶ Playback", padding=5)
        playback_frame.pack(side=tk.LEFT, padx=10)
        
        # Play/Pause button
        self.play_button = ttk.Button(
            playback_frame,
            text="▶ Play",
            command=self.toggle_playback,
            width=8
        )
        self.play_button.pack(side=tk.LEFT, padx=2)
        
        # Loop checkbox
        self.loop_var = tk.BooleanVar(value=False)
        self.loop_check = ttk.Checkbutton(
            playback_frame,
            text="Loop",
            variable=self.loop_var,
            command=self._update_loop
        )
        self.loop_check.pack(side=tk.LEFT, padx=5)
        
        # FPS controls
        fps_frame = ttk.Frame(playback_frame)
        fps_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(fps_frame, text="FPS:", font=("Arial", 9)).pack(side=tk.LEFT)
        
        self.fps_var = tk.IntVar(value=Config.DEFAULT_PLAYBACK_FPS)
        self.fps_spinbox = ttk.Spinbox(
            fps_frame,
            from_=Config.MIN_PLAYBACK_FPS,
            to=Config.MAX_PLAYBACK_FPS,
            width=5,
            textvariable=self.fps_var,
            command=self._update_fps
        )
        self.fps_spinbox.pack(side=tk.LEFT, padx=2)
        self.fps_spinbox.bind('<Return>', lambda e: self._update_fps())
        
        # === RIGHT SIDE: Skip and slider ===
        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.LEFT, padx=10)
        
        skip_frame = ttk.Frame(right_frame)
        skip_frame.pack(side=tk.TOP)
        
        ttk.Label(skip_frame, text="Skip:", font=("Arial", 9)).pack(side=tk.LEFT)
        
        self.skip_var = tk.IntVar(value=1)
        ttk.Spinbox(
            skip_frame,
            from_=1,
            to=100,
            width=5,
            textvariable=self.skip_var
        ).pack(side=tk.LEFT, padx=2)
        
        # Frame slider
        self.frame_slider = ttk.Scale(
            self,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self._slider_changed
        )
        self.frame_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

    def toggle_playback(self):
        """Toggle play/pause."""
        if self.app.playback_controller.toggle():
            self.play_button.config(text="⏸ Pause")
        else:
            self.play_button.config(text="▶ Play")

    def _update_fps(self):
        """Update playback FPS - FIXED: Real-time update."""
        try:
            fps = self.fps_var.get()
            self.app.playback_controller.set_fps(fps)
            # Update immediately if playing
            if self.app.playback_controller.is_playing():
                # Force immediate effect
                self.app.root.update_idletasks()
        except Exception as e:
            debug_log(f"Error updating FPS: {e}")

    def _update_loop(self):
        """Update loop setting."""
        self.app.playback_controller.set_loop(self.loop_var.get())

        

    def _slider_changed(self, value):
        """Handle slider value change - FIXED: Real-time preview."""
        if self.slider_updating or not self.app.has_frames():
            return
        
        # FIXED: Update immediately for real-time preview
        image_handler = self.app.image_handler
        if not image_handler or not image_handler.metadata:
            return
        
        max_idx = image_handler.metadata.total_frames - 1
        if max_idx <= 0:
            return
        
        frame_idx = int(float(value) / 100 * max_idx)
        
        # Cancel any pending debounced update
        if self.slider_update_pending is not None:
            self.after_cancel(self.slider_update_pending)
            self.slider_update_pending = None
        
        # Update immediately (no delay!)
        if frame_idx != self.app.current_frame_idx:
            self.app.set_current_frame(frame_idx)
    
    def _execute_slider_update(self, value):
        """Execute the actual frame update."""
        self.slider_update_pending = None
        
        if not self.app.has_frames():
            return
        
        image_handler = self.app.image_handler
        if not image_handler or not image_handler.metadata:
            return
        
        max_idx = image_handler.metadata.total_frames - 1
        if max_idx <= 0:
            return
        
        frame_idx = int(float(value) / 100 * max_idx)
        
        if frame_idx != self.app.current_frame_idx:
            self.app.set_current_frame(frame_idx)
    
    def update_slider_position(self, current_idx: int, total_frames: int):
        """Update slider position without triggering callback."""
        if total_frames <= 1:
            self.frame_slider.set(0)
            return
        
        self.slider_updating = True
        position = (current_idx / (total_frames - 1)) * 100
        self.frame_slider.set(position)
        self.slider_updating = False
    
    def update_frame_label(self, current_idx: int, total_frames: int):
        """Update frame counter label."""
        self.frame_label.config(text=f"Frame: {current_idx + 1}/{total_frames}")


# =============================================================================
# UI - COLLAPSIBLE PANEL (with tabs)
# =============================================================================

class CollapsiblePanel(ttk.Frame):
    """Collapsible panel with tabs for organizing controls."""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.is_collapsed = True
        self.current_tab = "Channels"
        
        self._create_ui()
    
    def _create_ui(self):
            
        self.header_frame = ttk.Frame(self)
        self.header_frame.pack(fill=tk.X, pady=(0, 0))
        
        self.collapse_button = ttk.Button(
            self.header_frame,
            text="▼ Settings (Click to Expand)",
            command=self.toggle_collapse,
            width=30
        )
        self.collapse_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.tab_frame = ttk.Frame(self.header_frame)
        self.tab_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.tab_buttons = {}
        tabs = ["Channels", "Enhancement", "Stats"]
        
        for tab_name in tabs:
            btn = ttk.Button(
                self.tab_frame,
                text=tab_name,
                command=lambda t=tab_name: self.switch_tab(t),
                width=12
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.tab_buttons[tab_name] = btn
        
        self.content_frame = ttk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
        
        self.canvas = tk.Canvas(
            self.content_frame,
            highlightthickness=0,
            height=300
        )
        
        self.scrollbar = ttk.Scrollbar(
            self.content_frame,
            orient="vertical",
            command=self.canvas.yview
        )
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw"
        )
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.bind(
            '<Configure>',
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width)
        )
        
        # FIXED: Remove bind_all and use local binding
        def on_mousewheel(event):
            # Only scroll if mouse is over the canvas
            widget = event.widget
            if widget == self.canvas or str(widget).startswith(str(self.canvas)):
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # FIXED: Bind only to this canvas, not globally
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        
        self.tab_contents = {}
        
        # Channels tab
        self.channels_container = ttk.Frame(self.scrollable_frame)
        self.tab_contents["Channels"] = self.channels_container
        self._create_channels_content()
        
        # Enhancement tab
        self.enhancement_container = ttk.Frame(self.scrollable_frame)
        self.tab_contents["Enhancement"] = self.enhancement_container
        self._create_enhancement_content()
        
        # Stats tab
        self.stats_container = ttk.Frame(self.scrollable_frame)
        self.tab_contents["Stats"] = self.stats_container
        self._create_stats_content()
        
        self._highlight_tab("Channels")


    def toggle_collapse(self):
        """Toggle between collapsed and expanded state."""
        if self.is_collapsed:
            self.expand()
        else:
            self.collapse()
    
    def expand(self):
        """Expand panel to show content with smooth resizing."""
        self.is_collapsed = False
        self.collapse_button.config(text="▲ Settings (Click to Collapse)")
        
        # FIXED: Use fill=tk.BOTH, expand=True for smooth expansion
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.switch_tab(self.current_tab)
        
        # Update window to accommodate content
        self.update_idletasks()
    
    def collapse(self):
        """Collapse panel to just show header."""
        self.is_collapsed = True
        self.collapse_button.config(text="▼ Settings (Click to Expand)")
        self.content_frame.pack_forget()
    
    def switch_tab(self, tab_name: str):
        """Switch to different tab."""
        if self.is_collapsed:
            self.expand()
        
        self.current_tab = tab_name
        
        for content in self.tab_contents.values():
            content.pack_forget()
        
        if tab_name in self.tab_contents:
            self.tab_contents[tab_name].pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self._highlight_tab(tab_name)
    
    def _highlight_tab(self, tab_name: str):
        """Highlight active tab button."""
        for name, btn in self.tab_buttons.items():
            if name == tab_name:
                btn.state(['pressed'])
            else:
                btn.state(['!pressed'])
    
    def _create_channels_content(self):
        """Create channel selector controls."""
        container = self.channels_container
        
        self.channel_indicator = ttk.Label(
            container,
            text="HIGH CONFIDENCE 🟢",
            font=("Arial", 12, "bold"),
            foreground="#00FF00",
            anchor=tk.CENTER
        )
        self.channel_indicator.pack(pady=(0, 15), fill=tk.X)
        
        # Channel list (will be populated dynamically)
        self.channel_list_frame = ttk.Frame(container)
        self.channel_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add channel button
        ttk.Button(
            container,
            text="➕ Add New Channel",
            command=self._add_channel_dialog
        ).pack(pady=10)
    
    def update_channel_list(self):
        """Update channel list display."""
        # Clear existing
        for widget in self.channel_list_frame.winfo_children():
            widget.destroy()
        
        channel_manager = self.app.session_state.channel_manager
        
        for channel in channel_manager.get_all_channels():
            self._create_channel_item(channel)
    
    def _create_channel_item(self, channel: ChannelDefinition):
        """Create single channel list item."""
        frame = ttk.Frame(self.channel_list_frame)
        frame.pack(fill=tk.X, pady=2)
        
        # Color indicator
        color_canvas = tk.Canvas(frame, width=20, height=20, highlightthickness=0)
        color_canvas.pack(side=tk.LEFT, padx=5)
        color_hex = '#{:02x}{:02x}{:02x}'.format(*channel.color_rgb)
        color_canvas.create_rectangle(0, 0, 20, 20, fill=color_hex, outline='')
        
        # Channel info
        info_frame = ttk.Frame(frame)
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        name_label = ttk.Label(info_frame, text=f"{channel.emoji} {channel.name}", font=("Arial", 10, "bold"))
        name_label.pack(anchor=tk.W)
        
        # Edit button
        ttk.Button(
            frame,
            text="Edit",
            command=lambda: self._edit_channel(channel),
            width=6
        ).pack(side=tk.RIGHT, padx=2)
        
        # Select button
        ttk.Button(
            frame,
            text="Select",
            command=lambda: self._select_channel(channel),
            width=8
        ).pack(side=tk.RIGHT, padx=2)
    
    def _select_channel(self, channel: ChannelDefinition):
        """Select channel as active."""
        self.app.set_active_channel(channel.id)
        
        color_hex = '#{:02x}{:02x}{:02x}'.format(*channel.color_rgb)
        self.channel_indicator.config(
            text=f"{channel.name.upper()} {channel.emoji}",
            foreground=color_hex
        )
    
    def _edit_channel(self, channel: ChannelDefinition):
        """Show dialog to edit channel properties."""
        dialog = tk.Toplevel(self.app.root)
        dialog.title(f"Edit Channel: {channel.name}")
        dialog.geometry("350x250")
        dialog.transient(self.app.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_reqwidth() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_reqheight() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name field
        ttk.Label(main_frame, text="Channel Name:", font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 5))
        name_var = tk.StringVar(value=channel.name)
        name_entry = ttk.Entry(main_frame, textvariable=name_var, width=30)
        name_entry.pack(fill=tk.X, pady=(0, 15))
        name_entry.focus()
        
        # Emoji field
        ttk.Label(main_frame, text="Emoji (optional):", font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 5))
        emoji_var = tk.StringVar(value=channel.emoji)
        emoji_entry = ttk.Entry(main_frame, textvariable=emoji_var, width=10)
        emoji_entry.pack(anchor=tk.W, pady=(0, 15))
        
        # Color preview
        color_frame = ttk.Frame(main_frame)
        color_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(color_frame, text="Color:", font=("Arial", 10)).pack(side=tk.LEFT)
        
        color_canvas = tk.Canvas(color_frame, width=50, height=30, highlightthickness=1)
        color_canvas.pack(side=tk.LEFT, padx=10)
        color_hex = '#{:02x}{:02x}{:02x}'.format(*channel.color_rgb)
        color_canvas.create_rectangle(0, 0, 50, 30, fill=color_hex, outline='black')
        
        def choose_color():
            from tkinter import colorchooser
            color = colorchooser.askcolor(color=color_hex, parent=dialog)
            if color and color[0]:
                rgb = tuple(int(c) for c in color[0])
                new_hex = '#{:02x}{:02x}{:02x}'.format(*rgb)
                color_canvas.delete("all")
                color_canvas.create_rectangle(0, 0, 50, 30, fill=new_hex, outline='black')
                color_canvas.rgb = rgb
        
        color_canvas.rgb = channel.color_rgb
        
        ttk.Button(color_frame, text="Choose Color", command=choose_color).pack(side=tk.LEFT, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def save_changes():
            new_name = name_var.get().strip()
            if not new_name:
                messagebox.showwarning("Invalid Name", "Channel name cannot be empty", parent=dialog)
                return
            
            new_emoji = emoji_var.get().strip() or "⚫"
            new_color = color_canvas.rgb
            
            # Update channel
            channel_manager = self.app.session_state.channel_manager
            channel_manager.update_channel(
                channel.id,
                name=new_name,
                emoji=new_emoji,
                color_rgb=new_color
            )
            
            # Refresh UI
            self.update_channel_list()
            
            # Update indicator if this is the active channel
            if self.app.active_channel_id == channel.id:
                self._select_channel(channel)
            
            dialog.destroy()
        
        ttk.Button(button_frame, text="Save", command=save_changes).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
    
    def _add_channel_dialog(self):
        """Show dialog to add new channel."""
        # Simple dialog
        name = tk.simpledialog.askstring("Add Channel", "Enter channel name:")
        if not name:
            return
        
        # For simplicity, use random color
        import random
        color_rgb = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        
        channel_manager = self.app.session_state.channel_manager
        channel = channel_manager.add_channel(name, color_rgb, "⚫")
        
        self.update_channel_list()
        self._select_channel(channel)
    
    def _create_enhancement_content(self):
        """Create enhancement filter controls."""
        container = self.enhancement_container
        
        left_col = ttk.Frame(container)
        left_col.grid(row=0, column=0, sticky="nsew", padx=5)
        
        right_col = ttk.Frame(container)
        right_col.grid(row=0, column=1, sticky="nsew", padx=5)
        
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        
        row = 0
        
        ttk.Label(left_col, text="Brightness:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.brightness_var = tk.IntVar(value=0)
        self.brightness_scale = ttk.Scale(
            left_col, from_=-100, to=100,
            variable=self.brightness_var,
            command=lambda x: self._update_filter('brightness'),
            length=150
        )
        self.brightness_scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        row += 1
        
        ttk.Label(left_col, text="Contrast:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.contrast_scale = ttk.Scale(
            left_col, from_=0.5, to=2.0,
            variable=self.contrast_var,
            command=lambda x: self._update_filter('contrast'),
            length=150
        )
        self.contrast_scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        row += 1
        
        ttk.Label(left_col, text="Gaussian Blur:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.gaussian_var = tk.IntVar(value=0)
        self.gaussian_scale = ttk.Scale(
            left_col, from_=0, to=15,
            variable=self.gaussian_var,
            command=lambda x: self._update_filter('gaussian_blur'),
            length=150
        )
        self.gaussian_scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        
        row = 0
        
        ttk.Label(right_col, text="Low Pass:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.lowpass_var = tk.IntVar(value=0)
        self.lowpass_scale = ttk.Scale(
            right_col, from_=0, to=100,
            variable=self.lowpass_var,
            command=lambda x: self._update_filter('low_pass_filter'),
            length=150
        )
        self.lowpass_scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        row += 1
        
        ttk.Label(right_col, text="High Pass:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.highpass_var = tk.IntVar(value=0)
        self.highpass_scale = ttk.Scale(
            right_col, from_=0, to=100,
            variable=self.highpass_var,
            command=lambda x: self._update_filter('high_pass_filter'),
            length=150
        )
        self.highpass_scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        row += 1
        
        self.temporal_var = tk.BooleanVar(value=False)
        self.temporal_check = ttk.Checkbutton(
            right_col,
            text="Temporal Smoothing",
            variable=self.temporal_var,
            command=lambda: self._update_filter('temporal_smoothing')
        )
        self.temporal_check.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        reset_btn = ttk.Button(container, text="Reset All Filters", command=self._reset_filters)
        reset_btn.grid(row=1, column=0, columnspan=2, pady=10)
        
        left_col.columnconfigure(1, weight=1)
        right_col.columnconfigure(1, weight=1)
    
    def _update_filter(self, filter_name: str):
        """Update filter and refresh display."""
        enhancer = self.app.enhancer
        
        if filter_name == 'brightness':
            enhancer.brightness = self.brightness_var.get()
        elif filter_name == 'contrast':
            enhancer.contrast = self.contrast_var.get()
        elif filter_name == 'gaussian_blur':
            enhancer.gaussian_blur = self.gaussian_var.get()
        elif filter_name == 'low_pass_filter':
            enhancer.low_pass_filter = self.lowpass_var.get()
        elif filter_name == 'high_pass_filter':
            enhancer.high_pass_filter = self.highpass_var.get()
        elif filter_name == 'temporal_smoothing':
            enhancer.temporal_smoothing = self.temporal_var.get()
            if not self.temporal_var.get():
                enhancer.reset_temporal_smoothing()
        
        self.app.schedule_display_update()
    
    def _reset_filters(self):
        """Reset all filters."""
        self.brightness_var.set(0)
        self.contrast_var.set(1.0)
        self.gaussian_var.set(0)
        self.lowpass_var.set(0)
        self.highpass_var.set(0)
        self.temporal_var.set(False)
        
        enhancer = self.app.enhancer
        enhancer.brightness = 0
        enhancer.contrast = 1.0
        enhancer.gaussian_blur = 0
        enhancer.low_pass_filter = 0
        enhancer.high_pass_filter = 0
        enhancer.temporal_smoothing = False
        enhancer.reset_temporal_smoothing()
        
        self.app.schedule_display_update()
    
    def _create_stats_content(self):
        """Create statistics display."""
        container = self.stats_container
        
        ttk.Label(container, text="Current Frame Statistics", font=("Arial", 11, "bold")).pack(pady=(0, 10))
        
        self.stats_text = tk.Text(
            container,
            height=10,
            wrap=tk.WORD,
            font=("Courier", 9),
            state=tk.DISABLED
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Button(
            container,
            text="Refresh Statistics",
            command=self.update_statistics
        ).pack(pady=10)
    
    def update_statistics(self):
        """Update statistics display."""
        if not self.app.has_frames():
            self._set_stats_text("No frames loaded")
            return
        
        annotation = self.app.get_current_annotation()
        
        if annotation is None or not annotation.has_annotations:
            self._set_stats_text("No annotations on current frame")
            return
        
        stats = f"Frame: {self.app.current_frame_idx + 1}\n"
        stats += "="*40 + "\n\n"
        
        total = annotation.total_labeled_pixels
        channel_manager = self.app.session_state.channel_manager
        
        for channel in channel_manager.get_all_channels():
            count = annotation.get_pixel_count(channel.id)
            if count > 0:
                pct = (count / total) * 100 if total > 0 else 0
                stats += f"{channel.emoji} {channel.name}:\n"
                stats += f"  {count:,} pixels ({pct:.1f}%)\n\n"
        
        stats += "-"*40 + "\n"
        stats += f"Total: {total:,} pixels\n"
        
        self._set_stats_text(stats)
    
    def _set_stats_text(self, text: str):
        """Set statistics text."""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', text)
        self.stats_text.config(state=tk.DISABLED)


# =============================================================================
# UI - WELCOME DIALOG (with Recent Sessions)
# =============================================================================

class WelcomeDialog:
    """Welcome dialog shown on startup with recent sessions."""
    
    def __init__(self, parent, recent_sessions_manager):
        self.parent = parent
        self.recent_sessions_manager = recent_sessions_manager
        self.result = None
        self.selected_file = None
        self.output_folder = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Welcome to {Config.APP_NAME} v{VERSION}")
        self.dialog.configure(bg=Config.COLOR_BG_PRIMARY)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.dialog.update_idletasks()
        width = 600
        height = 500
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        self.dialog.deiconify()
        self.dialog.lift()
        self.dialog.focus_force()
        
        self.setup_ui()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
    
    def setup_ui(self):
        """Create welcome dialog UI."""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = ttk.Label(
            title_frame,
            text=f"Welcome to {Config.APP_NAME}",
            font=("Arial", 18, "bold")
        )
        title.pack()
        
        version_label = ttk.Label(
            title_frame,
            text=f"{Config.APP_FULL_NAME} v{VERSION}",
            font=("Arial", 10)
        )
        version_label.pack()
        
        # Get Started section
        action_frame = ttk.LabelFrame(main_frame, text="Get Started", padding=15)
        action_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(
            action_frame,
            text="📁 Open TIF File",
            command=self.select_file,
            width=30
        ).pack(pady=5, fill=tk.X)
        
        ttk.Button(
            action_frame,
            text="📂 Load Session",
            command=self.load_session,
            width=30
        ).pack(pady=5, fill=tk.X)
        
        # Recent Sessions
        recent_frame = ttk.LabelFrame(main_frame, text="Recent Sessions", padding=10)
        recent_frame.pack(fill=tk.BOTH, expand=True)
        
        recent_sessions = self.recent_sessions_manager.get_recent_sessions(count=5)
        
        if recent_sessions:
            for session in recent_sessions:
                self._create_recent_session_item(recent_frame, session)
        else:
            ttk.Label(
                recent_frame,
                text="No recent sessions",
                foreground="#888"
            ).pack(pady=20)
        
        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.on_cancel,
            width=15
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            button_frame,
            text="About",
            command=self.show_about,
            width=15
        ).pack(side=tk.RIGHT)
    
    def _create_recent_session_item(self, parent, session: Dict):
        """Create single recent session list item."""
        frame = ttk.Frame(parent, relief=tk.RAISED, borderwidth=1)
        frame.pack(fill=tk.X, pady=3)
        
        btn = ttk.Button(
            frame,
            text=f"{session['name']}\n{session['labeled_frames']} frames labeled • {self._format_time_ago(session['last_opened'])}",
            command=lambda: self._load_recent_session(session)
        )
        btn.pack(fill=tk.X, padx=5, pady=5)
    
    def _format_time_ago(self, iso_datetime: str) -> str:
        """Format datetime as relative time."""
        try:
            dt = datetime.datetime.fromisoformat(iso_datetime)
            now = datetime.datetime.now()
            delta = now - dt
            
            if delta.days > 365:
                return f"{delta.days // 365} year(s) ago"
            elif delta.days > 30:
                return f"{delta.days // 30} month(s) ago"
            elif delta.days > 0:
                return f"{delta.days} day(s) ago"
            elif delta.seconds > 3600:
                return f"{delta.seconds // 3600} hour(s) ago"
            elif delta.seconds > 60:
                return f"{delta.seconds // 60} minute(s) ago"
            else:
                return "Just now"
        except:
            return "Unknown"
    
    def select_file(self):
        """Open file selection dialog."""
        file_path = filedialog.askopenfilename(
            title="Select TIF File",
            filetypes=[("TIF files", "*.tif;*.tiff"), ("All files", "*.*")],
            parent=self.dialog
        )
        
        if file_path:
            self.selected_file = file_path
            
            if messagebox.askyesno(
                "Output Folder",
                "Would you like to set an output folder now?",
                parent=self.dialog
            ):
                folder = filedialog.askdirectory(
                    title="Select Output Folder",
                    parent=self.dialog
                )
                if folder:
                    self.output_folder = folder
            
            self.result = 'file_selected'
            self.dialog.destroy()
    
    def load_session(self):
        """Load existing session with robust path detection."""
        folder = filedialog.askdirectory(
            title="Select Session Folder (the folder containing session.dat)",
            parent=self.dialog
        )
        
        if not folder:
            return
        
        folder_path = Path(folder)
        session_dat = folder_path / Config.SESSION_LOG_BINARY
        
        # Check if selected folder contains session.dat
        if session_dat.exists():
            self.selected_file = folder
            self.result = 'session_selected'
            self.dialog.destroy()
            return
        
        # If not, check immediate subfolders for session folders
        session_folders = []
        try:
            for subfolder in folder_path.iterdir():
                if subfolder.is_dir():
                    sub_session_dat = subfolder / Config.SESSION_LOG_BINARY
                    if sub_session_dat.exists():
                        session_folders.append(subfolder)
        except Exception as e:
            log(f"Error scanning subfolders: {e}", 'WARNING')
        
        if len(session_folders) == 1:
            # Found exactly one session folder
            if messagebox.askyesno(
                "Session Found",
                f"Found session folder:\n{session_folders[0].name}\n\nLoad this session?",
                parent=self.dialog
            ):
                self.selected_file = str(session_folders[0])
                self.result = 'session_selected'
                self.dialog.destroy()
                return
        elif len(session_folders) > 1:
            # Multiple session folders found - let user choose
            choice_dialog = tk.Toplevel(self.dialog)
            choice_dialog.title("Select Session")
            choice_dialog.geometry("400x300")
            choice_dialog.transient(self.dialog)
            choice_dialog.grab_set()
            
            ttk.Label(
                choice_dialog,
                text=f"Found {len(session_folders)} session folders. Select one:",
                font=("Arial", 10)
            ).pack(pady=10)
            
            listbox = tk.Listbox(choice_dialog, height=10)
            listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            for sf in session_folders:
                listbox.insert(tk.END, sf.name)
            
            selected_session = [None]
            
            def on_select():
                selection = listbox.curselection()
                if selection:
                    selected_session[0] = session_folders[selection[0]]
                    choice_dialog.destroy()
            
            ttk.Button(choice_dialog, text="Load Selected", command=on_select).pack(pady=5)
            
            choice_dialog.wait_window()
            
            if selected_session[0]:
                self.selected_file = str(selected_session[0])
                self.result = 'session_selected'
                self.dialog.destroy()
                return
        
        # No session found - offer to select file directly
        if messagebox.askyesno(
            "Session Not Found",
            f"Could not find '{Config.SESSION_LOG_BINARY}' in:\n{folder_path}\n\n"
            f"Would you like to select the session.dat file directly?",
            parent=self.dialog
        ):
            file_path = filedialog.askopenfilename(
                title=f"Select {Config.SESSION_LOG_BINARY} file",
                filetypes=[("Session files", "*.dat"), ("All files", "*.*")],
                initialdir=folder,
                parent=self.dialog
            )
            
            if file_path and Path(file_path).name == Config.SESSION_LOG_BINARY:
                # Use the parent folder of the session.dat file
                self.selected_file = str(Path(file_path).parent)
                self.result = 'session_selected'
                self.dialog.destroy()
            elif file_path:
                messagebox.showerror(
                    "Invalid File",
                    f"Please select a file named '{Config.SESSION_LOG_BINARY}'",
                    parent=self.dialog
                )
    
    def _load_recent_session(self, session: Dict):
        """Load session from recent list."""
        path = self.recent_sessions_manager.get_session_path(session)
        
        if path:
            self.selected_file = str(path)
            self.result = 'session_selected'
            self.dialog.destroy()
        else:
            messagebox.showerror(
                "Session Not Found",
                f"Cannot find session folder:\n{session['name']}",
                parent=self.dialog
            )
    
    def show_about(self):
        """Show about dialog."""
        features_text = "\n• ".join([""] + Config.APP_FEATURES[:8]) 
        messagebox.showinfo(
            "About",
            f"{Config.APP_NAME} - {Config.APP_FULL_NAME}\n"
            f"Version {VERSION}\n"
            f"Build: {BUILD_DATE}\n\n"
            f"Key Features:{features_text}\n\n"
            "Professional-grade annotation tool for large scientific images.",
            parent=self.dialog
        )
    
    def on_cancel(self):
        """Handle cancel."""
        if messagebox.askyesno(
            "Exit",
            f"Exit {Config.APP_NAME}?",
            parent=self.dialog
        ):
            self.result = 'cancelled'
            self.dialog.destroy()
    
    def show(self):
        """Show dialog and wait for result."""
        self.dialog.update()
        self.dialog.update_idletasks()
        self.dialog.focus_force()
        self.dialog.wait_window()
        return self.result, self.selected_file, self.output_folder


# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class MATApplication:
    """Main application class integrating all components."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        
        # Core components
        self.image_handler: Optional[TIFHandler] = None
        self.enhancer = ImageEnhancer()
        self.session_manager: Optional[SessionManager] = None
        self.file_manager = FileManager(self)
        self.memory_manager = MemoryManager()
        self.recent_sessions = RecentSessionsManager(max_sessions=Config.MAX_RECENT_SESSIONS)
        
        # State
        self.session_state: Optional[SessionState] = None
        self.current_frame_idx = 0
        self.current_tool = ToolType.DRAW
        self.brush_size = Config.BRUSH_SIZE_DEFAULT
        self.output_dir: Optional[Path] = None
        self.frame_annotations: Dict[int, FrameAnnotation] = {}
        # FIXED: Opacity initialization 
        self.mask_opacity = Config.CHANNEL_OPACITY  # Default 0.4
        
        # Active channel tracking
        self.active_channel_id: Optional[str] = None
        
        # Display state
        self.display_info: Optional[Tuple[int, int, int, int]] = None
        self.undo_stack = UndoStack(max_size=Config.UNDO_STACK_SIZE)
        
        # Propagation
        self.propagation_engine: Optional[PropagationEngine] = None
        self.propagation_confidence: float = 0.0
        
        # Update control
        self.update_pending = False
        self.perf_monitor = PerformanceMonitor()

        # Playback controller for frame animation
        self.playback_controller = PlaybackController(self)
        
        # Frame save state tracking
        self.frame_modified: Dict[int, bool] = {}  # Track which frames are modified since last save
        
        # Register cleanup callback
        self.memory_manager.register_cleanup_callback(self._cleanup_old_caches)

        # Contour mode selection
        self.contour_mode_var = None
        
        # Project info
        self.project_name: str = None
        self.project_dir: Optional[Path] = None
        self.original_tif_path: Optional[Path] = None

        # Setup UI
        self.setup_theme()
        self.setup_ui()
        self.setup_menu()
        self.setup_bindings()

        # FIXED: Synchronize opacity UI with initial value
        if hasattr(self, 'opacity_var'):
            self.opacity_var.set(self.mask_opacity * 100)
            if hasattr(self, 'opacity_label'):
                self.opacity_label.config(text=f"{int(self.mask_opacity * 100)}%")
        
        # Start monitoring
        self.start_memory_monitoring()
        
        log(f"{Config.APP_NAME} initialized")
    
    def setup_theme(self):
        """Apply dark theme to application."""
        self.root.configure(bg=Config.COLOR_BG_PRIMARY)
        
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
            style.configure("TFrame", background=Config.COLOR_BG_PRIMARY)
            style.configure("TButton", background=Config.COLOR_BG_SECONDARY, foreground=Config.COLOR_FG_PRIMARY)
            style.configure("TLabel", background=Config.COLOR_BG_PRIMARY, foreground=Config.COLOR_FG_PRIMARY)
            style.configure("TLabelframe", background=Config.COLOR_BG_PRIMARY, foreground=Config.COLOR_FG_PRIMARY)
            style.configure("TLabelframe.Label", background=Config.COLOR_BG_PRIMARY, foreground=Config.COLOR_FG_PRIMARY)
        except Exception as e:
            debug_log(f"Error setting theme: {e}")
    
    def setup_ui(self):
        """Create all UI components."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Toolbar
        self.toolbar = ToolbarManager(self.main_frame, self)
        self.toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # MAIN VERTICAL SPLIT (top: canvas+controls, bottom: navigation+settings)
        self.main_vertical_paned = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.main_vertical_paned.pack(fill=tk.BOTH, expand=True)
        
        # TOP SECTION: Horizontal split (canvas + controls)
        top_section = ttk.Frame(self.main_vertical_paned)
        self.main_vertical_paned.add(top_section, weight=4)
        
        self.paned_window = ttk.PanedWindow(top_section, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Canvas
        canvas_container = ttk.Frame(self.paned_window)
        self.paned_window.add(canvas_container, weight=4)
        
        self.canvas = OptimizedCanvas(canvas_container, self)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Controls
        controls_container = ttk.Frame(self.paned_window)
        self.paned_window.add(controls_container, weight=1)
        
        self.setup_controls(controls_container)
        
        # BOTTOM SECTION: Navigation + Settings (resizable)
        bottom_section = ttk.Frame(self.main_vertical_paned)
        self.main_vertical_paned.add(bottom_section, weight=1)
        
        # Navigation
        self.navigation = NavigationPanel(bottom_section, self)
        self.navigation.pack(fill=tk.X, pady=5)
        
        # Collapsible panel
        self.collapsible_panel = CollapsiblePanel(bottom_section, self)
        self.collapsible_panel.pack(fill=tk.X, pady=5)
    
    def setup_controls(self, parent):
        """Setup control buttons panel with scrolling."""
        # CREATE SCROLLABLE CONTAINER
        canvas_container = ttk.Frame(parent)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        controls_canvas = tk.Canvas(canvas_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=controls_canvas.yview)
        
        # Scrollable frame
        controls = ttk.Frame(controls_canvas, padding=10)
        
        # Configure scrolling
        controls_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        canvas_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        
        # Update scroll region when controls change size
        def configure_scroll_region(event=None):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))
        
        controls.bind("<Configure>", configure_scroll_region)
        
        # Bind canvas width to frame width
        def configure_canvas_width(event):
            controls_canvas.itemconfig(canvas_window, width=event.width)
        
        controls_canvas.bind("<Configure>", configure_canvas_width)
        
        # Bind mousewheel
        def on_mousewheel(event):
            controls_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        controls_canvas.bind("<Enter>", lambda e: controls_canvas.bind_all("<MouseWheel>", on_mousewheel))
        controls_canvas.bind("<Leave>", lambda e: controls_canvas.unbind_all("<MouseWheel>"))
        
        # === NOW ADD ALL THE CONTROLS ===
        
        ttk.Label(controls, text="Tools", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        ttk.Button(controls, text="Draw (D)", command=lambda: self.set_tool(ToolType.DRAW)).pack(pady=2, fill=tk.X)
        ttk.Button(controls, text="Erase (E)", command=lambda: self.set_tool(ToolType.ERASE)).pack(pady=2, fill=tk.X)
        ttk.Button(controls, text="Erase All Channels (A)", command=lambda: self.set_tool(ToolType.ERASE_ADMIN)).pack(pady=2, fill=tk.X)
        ttk.Button(controls, text="Contour Draw (X)", command=lambda: self.set_tool(ToolType.CONTOUR)).pack(pady=2, fill=tk.X)
        
        # === NEW: CONTOUR MODE SELECTION ===
        contour_mode_frame = ttk.LabelFrame(controls, text="Contour Mode", padding=10)
        contour_mode_frame.pack(fill=tk.X, pady=5)
        
        self.contour_mode_var = tk.StringVar(value="fill")
        
        ttk.Radiobutton(
            contour_mode_frame,
            text="Fill Contour",
            variable=self.contour_mode_var,
            value="fill"
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            contour_mode_frame,
            text="Outline Only",
            variable=self.contour_mode_var,
            value="outline"
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            contour_mode_frame,
            text="Ask Each Time",
            variable=self.contour_mode_var,
            value="ask"
        ).pack(anchor=tk.W)
        # === END CONTOUR MODE ===
        
        ttk.Button(controls, text="Clear Frame (C)", command=self.clear_mask).pack(pady=2, fill=tk.X)
        ttk.Button(controls, text="⚠️ Clear All Frames", command=self.clear_all_masks).pack(pady=2, fill=tk.X)
        ttk.Button(controls, text="Undo (Ctrl+Z)", command=self.undo_action).pack(pady=2, fill=tk.X)
        
        ttk.Separator(controls, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Propagation section
        prop_frame = ttk.LabelFrame(controls, text="⚡ Propagation", padding=10)
        prop_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(prop_frame, text="⬅️ Back (B)", command=self.propagate_back).pack(pady=2, fill=tk.X)
        ttk.Button(prop_frame, text="Next ➡️ (P)", command=self.propagate_next).pack(pady=2, fill=tk.X)
        
        self.confidence_label = ttk.Label(prop_frame, text="", font=("Arial", 9))
        self.confidence_label.pack(pady=5)
        
        ttk.Separator(controls, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Opacity control
        opacity_frame = ttk.LabelFrame(controls, text="Mask Opacity", padding=10)
        opacity_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(opacity_frame, text="Opacity:", font=("Arial", 10)).pack(anchor=tk.W)
        
        self.opacity_var = tk.DoubleVar(value=40)
        self.opacity_scale = ttk.Scale(
            opacity_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.opacity_var,
            command=self._opacity_slider_changed  # FIXED: Use dedicated method
        )
        self.opacity_scale.pack(fill=tk.X, pady=5)
        
        self.opacity_label = ttk.Label(opacity_frame, text="40%", font=("Arial", 9))
        self.opacity_label.pack()
        
        ttk.Label(opacity_frame, text="[ and ] keys", font=("Arial", 8), foreground="#888").pack(pady=(5, 0))
        
        ttk.Separator(controls, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # View controls
        ttk.Label(controls, text="View", font=("Arial", 11, "bold")).pack(pady=(5, 5))
        ttk.Button(controls, text="Zoom In (+)", command=self.zoom_in).pack(pady=2, fill=tk.X)
        ttk.Button(controls, text="Zoom Out (-)", command=self.zoom_out).pack(pady=2, fill=tk.X)
        ttk.Button(controls, text="Reset View", command=self.reset_view).pack(pady=2, fill=tk.X)
        
        ttk.Separator(controls, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Export
        ttk.Label(controls, text="Export", font=("Arial", 11, "bold")).pack(pady=(5, 5))
        ttk.Button(controls, text="Save Current", command=self.save_current).pack(pady=2, fill=tk.X)
        ttk.Button(controls, text="Save All", command=self.save_all).pack(pady=2, fill=tk.X)

    def _opacity_slider_changed(self, value):
        """Handle opacity slider change with proper type conversion."""
        try:
            # CRITICAL: Force float conversion (slider passes string!)
            float_value = float(value)
            self.update_opacity(float_value)
        except (ValueError, TypeError) as e:
            debug_log(f"Error in opacity slider: {e}")

    def setup_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open TIF...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Set Output Folder", command=self.set_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Save Current", command=self.save_current, accelerator="Ctrl+S")
        file_menu.add_command(label="Save All", command=self.save_all)
        file_menu.add_separator()
        file_menu.add_command(label="Load Session...", command=self.load_session_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_application)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=self.zoom_in, accelerator="+")
        view_menu.add_command(label="Zoom Out", command=self.zoom_out, accelerator="-")
        view_menu.add_command(label="Reset View", command=self.reset_view)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_bindings(self):
        """Setup keyboard shortcuts."""
        self.root.bind("<Control-o>", lambda e: self.open_file())
        # Playback controls
        self.root.bind("<space>", lambda e: self.navigation.toggle_playback())
        self.root.bind("<Return>", lambda e: self.navigation.toggle_playback())
        
        self.root.bind("<Control-s>", lambda e: self.save_current())
        self.root.bind("<Control-z>", lambda e: self.undo_action())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<plus>", lambda e: self.zoom_in())
        self.root.bind("<minus>", lambda e: self.zoom_out())
        self.root.bind("d", lambda e: self.set_tool(ToolType.DRAW))
        self.root.bind("e", lambda e: self.set_tool(ToolType.ERASE))
        self.root.bind("a", lambda e: self.set_tool(ToolType.ERASE_ADMIN))
        self.root.bind("x", lambda e: self.set_tool(ToolType.CONTOUR))
        self.root.bind("c", lambda e: self.clear_mask())
        self.root.bind("p", lambda e: self.propagate_next())
        self.root.bind("b", lambda e: self.propagate_back())
        self.root.bind("[", lambda e: self.decrease_opacity())
        self.root.bind("]", lambda e: self.increase_opacity())
    
    def start_memory_monitoring(self):
        """Start periodic memory monitoring."""
        try:
            mem_mb = self.perf_monitor.get_memory_usage_mb()
            self.toolbar.update_memory_display(mem_mb)
            
            self.memory_manager.auto_cleanup_if_needed()
            
            self.root.after(Config.CLEANUP_INTERVAL_MS, self.start_memory_monitoring)
        except Exception as e:
            debug_log(f"Error in memory monitoring: {e}")
    
    def _cleanup_old_caches(self):
        """Cleanup callback: remove old data from memory."""
        if self.image_handler:
            self.image_handler.clear_cache()

    def update_title(self):
        """Update window title with project name."""
        if self.project_name:
            self.root.title(f"{Config.APP_NAME} v{VERSION} - {self.project_name}")
        else:
            self.root.title(f"{Config.APP_NAME} v{VERSION}")
    
    def has_frames(self) -> bool:
        """Check if frames are loaded."""
        return self.image_handler is not None and self.image_handler.metadata is not None
    
    def save_session(self) -> bool:
        """
        Save current session state.
        
        FIXED: This is now a method of MATApplication (not SessionManager).
        """
        try:
            if not self.has_frames():
                messagebox.showwarning(
                    "No Project",
                    "No project loaded. Please open a TIF file first.",
                    parent=self.root
                )
                return False
            
            if not self.project_dir:
                messagebox.showwarning(
                    "No Project",
                    "No project folder created yet. Please save at least one frame first.",
                    parent=self.root
                )
                return False
            
            log("Saving session...")
            
            # Create session state from current application state
            session_state = SessionState(
                project_name=self.project_name,
                tif_path=str(self.original_tif_path) if self.original_tif_path else "",
                project_dir=str(self.project_dir),
                current_frame_index=self.current_frame_idx,
                total_frames=self.image_handler.metadata.total_frames,
                image_metadata=self.image_handler.metadata,
                channel_manager=self.session_state.channel_manager if self.session_state else ChannelManager(),
                active_channel_id=self.active_channel_id
            )
            
            # Copy ALL annotations
            session_state.annotations = {}
            for frame_idx, annotation in self.frame_annotations.items():
                session_state.annotations[frame_idx] = annotation
            
            debug_log(f"Saving {len(session_state.annotations)} frame annotations")
            
            # Store view state
            session_state.zoom_factor = self.canvas.zoom_factor
            session_state.pan_x = self.canvas.pan_x
            session_state.pan_y = self.canvas.pan_y
            
            # Store tool state
            session_state.current_tool = self.current_tool
            session_state.brush_size = self.brush_size
            
            # Store opacity
            session_state.mask_opacity = self.mask_opacity
            
            # Store enhancement settings
            session_state.brightness = self.enhancer.brightness
            session_state.contrast = self.enhancer.contrast
            session_state.gaussian_blur = self.enhancer.gaussian_blur
            session_state.low_pass_filter = self.enhancer.low_pass_filter
            session_state.high_pass_filter = self.enhancer.high_pass_filter
            session_state.temporal_smoothing = self.enhancer.temporal_smoothing
            
            # Update labeled frames set
            session_state.labeled_frames = set()
            for frame_idx, annotation in session_state.annotations.items():
                if annotation.has_annotations:
                    session_state.labeled_frames.add(frame_idx)
            
            debug_log(f"Session has {len(session_state.labeled_frames)} labeled frames")
            
            # Initialize session manager if needed
            if not self.session_manager:
                self.session_manager = SessionManager(self.project_dir)
            
            # Save to file using SessionManager
            success = self.session_manager.save_session_to_file(session_state)
            
            if success:
                log("Session saved successfully")
                
                # Update recent sessions
                self.recent_sessions.add_session(self.project_dir, session_state)
                
                return True
            else:
                messagebox.showerror(
                    "Save Error",
                    "Failed to save session",
                    parent=self.root
                )
                return False
            
        except Exception as e:
            log(f"Error saving session: {e}", 'ERROR')
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "Save Error",
                f"Failed to save session:\n{str(e)}",
                parent=self.root
            )
            return False
    
    def get_current_annotation(self) -> Optional[FrameAnnotation]:
        """Get or create annotation for current frame."""
        if not self.has_frames():
            return None
        
        frame_idx = self.current_frame_idx
        
        if frame_idx not in self.frame_annotations:
            self.frame_annotations[frame_idx] = FrameAnnotation(frame_index=frame_idx)
            # FIXED: Initialize frame_modified state
            if frame_idx not in self.frame_modified:
                self.frame_modified[frame_idx] = False
        
        return self.frame_annotations[frame_idx]
    

    def get_current_channel_mask(self) -> Optional[np.ndarray]:
        """Get mask for current frame and active channel - WITH AUTO-CREATION."""
        annotation = self.get_current_annotation()
        
        if annotation is None:
            debug_log(f"No annotation for frame {self.current_frame_idx}")
            return None
        
        if not self.active_channel_id:
            debug_log(f"No active channel selected")
            return None
        
        mask = annotation.get_mask(self.active_channel_id)

        # AUTO-CREATE mask if needed
        if mask is None:
            log(f"Creating new mask for channel {self.active_channel_id} on frame {self.current_frame_idx}")
            mask = self.image_handler.create_empty_mask(for_display=True)
            annotation.set_mask(self.active_channel_id, mask)

        return mask

    def set_active_channel(self, channel_id: str):
        """Set active annotation channel."""
        self.active_channel_id = channel_id
        self.schedule_display_update()
    
    def open_file(self):
        """Open TIF file dialog."""
        try:
            file_path = filedialog.askopenfilename(
                title="Open TIF File",
                filetypes=[("TIF files", "*.tif;*.tiff"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            self.load_file(file_path)
            
        except Exception as e:
            log(f"Error opening file: {e}", 'ERROR')
            messagebox.showerror("Error", f"Failed to open file: {str(e)}")
    
    def load_file(self, file_path: str):
        """Load TIF file and initialize session."""
        progress = None
        try:
            progress = ProgressDialog(self.root, "Loading TIF File")
            progress.update_progress(0, 100, "Analyzing file...")
            
            self.image_handler = TIFHandler(file_path)
            
            progress.update_progress(50, 100, "Initializing session...")
            
            # Create session state with channel manager
            channel_manager = ChannelManager()
            channel_manager.load_default_channels()

            # Set project info
            self.project_name = Path(file_path).stem
            self.original_tif_path = Path(file_path)
            
            self.session_state = SessionState(
                project_name=Path(file_path).stem,
                created_at=datetime.datetime.now().isoformat(),
                last_saved=datetime.datetime.now().isoformat(),
                image_metadata=self.image_handler.metadata,  # ADDED: Set image_metadata
                channel_manager=channel_manager
            )
            # Set first channel as active
            first_channel = channel_manager.get_all_channels()[0]
            self.active_channel_id = first_channel.id
            
            # Update channel list UI
            self.collapsible_panel.update_channel_list()
            self.collapsible_panel._select_channel(first_channel)
            
            # Clear old data
            self.frame_annotations.clear()
            self.undo_stack.clear()
            self.current_frame_idx = 0
            
            # Initialize propagation engine
            self.propagation_engine = PropagationEngine(self.image_handler)
            
            progress.update_progress(90, 100, "Updating display...")
            
            self.update_frame_info()
            self.update_display_now()
            
            progress.close()
            
            messagebox.showinfo(
                "File Loaded",
                f"Successfully loaded!\n\n"
                f"Frames: {self.image_handler.metadata.total_frames}\n"
                f"Resolution: {self.image_handler.metadata.original_width}×{self.image_handler.metadata.original_height}\n"
                f"Channels: {len(channel_manager.get_all_channels())}"
            )
            
            log(f"Loaded {file_path}")
            
        except Exception as e:
            if progress:
                progress.close()
            log(f"Error loading file: {e}", 'ERROR')
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def set_output_folder(self):
        """Set output folder for saving."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_dir = Path(folder)
            messagebox.showinfo("Output Folder", f"Output folder set to:\n{folder}")
            log(f"Output folder: {folder}")
    
    def save_current(self):
        """Save current frame."""
        if not self.has_frames():
            messagebox.showwarning("Warning", "No frames loaded")
            return
        
        if not self.output_dir:
            self.set_output_folder()
            if not self.output_dir:
                return
        
        self.file_manager.save_current_frame(self.output_dir)
        
        # Mark frame as saved
        self.frame_modified[self.current_frame_idx] = False
        self.update_frame_info()
    
    def save_all(self):
        """Save all labeled frames."""
        if not self.has_frames():
            messagebox.showwarning("Warning", "No frames loaded")
            return
        
        labeled_frames = [idx for idx, ann in self.frame_annotations.items() if ann.has_annotations]
        
        if not labeled_frames:
            messagebox.showinfo("Info", "No labeled frames to save")
            return
        
        if not self.output_dir:
            self.set_output_folder()
            if not self.output_dir:
                return
        
        # Confirm
        response = messagebox.askyesno(
            "Save All Frames",
            f"This will save {len(labeled_frames)} labeled frames.\n\n"
            "Continue?",
            parent=self.root
        )
        
        if not response:
            return
        
        progress = ProgressDialog(self.root, "Saving All Frames")
        
        saved_count = 0
        failed_count = 0
        failed_frames = []
        
        image_handler = self.image_handler
        base_name = image_handler.metadata.file_path.stem
        frame_indexing = image_handler.metadata.frame_indexing
        
        # Ensure project structure exists
        if self.file_manager.current_project_dir is None:
            folders = self.file_manager.create_multichannel_structure(
                self.output_dir, base_name, image_handler.metadata.file_path
            )
            if folders is None:
                progress.close()
                messagebox.showerror("Error", "Failed to create project structure")
                return
        else:
            folders = self.file_manager._get_existing_folders()
        
        channel_manager = self.session_state.channel_manager
        
        try:
            for i, frame_idx in enumerate(labeled_frames):
                # Update progress
                progress.update_progress(
                    i, len(labeled_frames),
                    f"Saving frame {frame_idx + 1}...",
                    f"{saved_count} saved, {failed_count} failed"
                )
                
                try:
                    # Load original frame
                    original_frame = image_handler.load_frame(frame_idx, for_display=False)
                    if original_frame is None:
                        failed_count += 1
                        failed_frames.append(frame_idx)
                        continue
                    
                    annotation = self.frame_annotations[frame_idx]
                    filename = self.file_manager.get_frame_filename(base_name, frame_idx, frame_indexing)
                    
                    # Save original image
                    img_path = folders['image'] / filename
                    pil_frame = Image.fromarray(original_frame)
                    pil_frame.save(img_path, compression=Config.EXPORT_COMPRESSION)
                    
                    # Create combined mask (binary 255 for ANY annotation)
                    h, w = image_handler.metadata.original_height, image_handler.metadata.original_width
                    combined_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # Save each channel mask
                    for channel in channel_manager.get_all_channels():
                        display_mask = annotation.get_mask(channel.id)
                        if display_mask is not None and np.any(display_mask):
                            full_mask = image_handler.upscale_mask(display_mask)
                            mask_path = folders[channel.id] / filename
                            pil_mask = Image.fromarray(full_mask)
                            pil_mask.save(mask_path, compression=Config.EXPORT_COMPRESSION)
                            
                            # Add to combined mask (binary 255)
                            combined_mask[full_mask > 0] = 255
                    
                    # Save combined mask to Mask_All
                    if np.any(combined_mask):
                        mask_all_path = folders['mask_all'] / filename
                        pil_combined = Image.fromarray(combined_mask)
                        pil_combined.save(mask_all_path, compression=Config.EXPORT_COMPRESSION)
                    
                    # Create composite overlay
                    overlay_frame = original_frame.copy()
                    if len(overlay_frame.shape) == 2:
                        overlay_frame = np.stack([overlay_frame] * 3, axis=-1)
                    
                    for channel in reversed(channel_manager.get_all_channels()):
                        display_mask = annotation.get_mask(channel.id)
                        if display_mask is not None and np.any(display_mask):
                            full_mask = image_handler.upscale_mask(display_mask)
                            mask_pixels = np.where(full_mask > 0)
                            if len(mask_pixels[0]) > 0:
                                overlay_frame[mask_pixels] = channel.color_rgb
                    
                    overlay_path = folders['overlay'] / filename
                    pil_overlay = Image.fromarray(overlay_frame)
                    pil_overlay.save(overlay_path, compression=Config.EXPORT_COMPRESSION)
                    
                    saved_count += 1
                    
                except Exception as e:
                    log(f"Error saving frame {frame_idx}: {e}", 'ERROR')
                    failed_count += 1
                    failed_frames.append(frame_idx)
            
            # Final progress update
            progress.update_progress(
                len(labeled_frames), len(labeled_frames),
                "Complete!",
                f"{saved_count} saved, {failed_count} failed"
            )
            
            # Auto-save session (now saves to project folder)
            # Auto-save session
            if Config.AUTOSAVE_ON_FRAME_SAVE and self.file_manager.current_project_dir:
                self.project_dir = self.file_manager.current_project_dir
                self.save_session()
            
            # Mark all successfully saved frames as saved
            for frame_idx in labeled_frames:
                if frame_idx not in failed_frames:
                    self.frame_modified[frame_idx] = False
            
            progress.close()
            
            # Update display if on a frame that was saved
            self.update_frame_info()
            
            # Show summary
            summary = f"Save All Complete!\n\n"
            summary += f"Successfully saved: {saved_count}/{len(labeled_frames)} frames\n"
            
            if failed_count > 0:
                summary += f"\nFailed frames: {failed_count}\n"
                summary += f"Frame indices: {failed_frames[:10]}"
                if len(failed_frames) > 10:
                    summary += f"... and {len(failed_frames) - 10} more"
                messagebox.showwarning("Save Complete (with errors)", summary, parent=self.root)
            else:
                messagebox.showinfo("Save Complete", summary, parent=self.root)
            
            log(f"Save All: {saved_count} frames saved, {failed_count} failed")
            
        except Exception as e:
            progress.close()
            log(f"Error in Save All: {e}", 'ERROR')
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Save All failed:\n{str(e)}")
    
    
    def load_session_dialog(self):
        """Load existing session dialog."""
        folder = filedialog.askdirectory(title="Select Session Folder")
        if not folder:
            return
        
        session_dir = Path(folder)
        
        if not (session_dir / Config.SESSION_LOG_BINARY).exists():
            messagebox.showerror(
                "Invalid Session",
                f"Not a valid session folder:\n{Config.SESSION_LOG_BINARY} not found"
            )
            return
        
        self.load_session_with_fallback(session_dir)
    
    def load_session_with_fallback(self, session_dir: Path) -> bool:
        """Load session with automatic TIF loading and fallback."""
        try:
            # CRITICAL FIX: Ensure we have a Path object and resolve it
            if isinstance(session_dir, str):
                session_dir = Path(session_dir)
            
            session_dir = session_dir.resolve()  # Get absolute path
            
            log(f"Attempting to load session from: {session_dir}")
            
            # Check if session directory exists
            if not session_dir.exists():
                error_msg = f"Session folder not found:\\n{session_dir}"
                log(error_msg, 'ERROR')
                messagebox.showerror("Error", error_msg, parent=self.root)
                return False
            
            if not session_dir.is_dir():
                error_msg = f"Path is not a directory:\n{session_dir}"
                log(error_msg, 'ERROR')
                messagebox.showerror("Error", error_msg, parent=self.root)
                return False
            
            # Check if session.dat exists
            session_dat_path = session_dir / Config.SESSION_LOG_BINARY
            
            log(f"Checking for session file: {session_dat_path}")
            log(f"  Session.dat exists: {session_dat_path.exists()}")
            log(f"  Session dir exists: {session_dir.exists()}")
            log(f"  Session dir is_dir: {session_dir.is_dir()}")
            
            if not session_dat_path.exists():
                # List what's actually in the directory
                try:
                    contents = list(session_dir.iterdir())
                    contents_str = "\n".join([f"  - {f.name}" for f in contents[:10]])
                    if len(contents) > 10:
                        contents_str += f"\n  ... and {len(contents) - 10} more files"
                except Exception as e:
                    contents_str = f"  (Could not list directory: {e})"
                
                error_msg = (
                    f"Session file not found:\n{session_dat_path}\n\n"
                    f"Directory contents:\n{contents_str}"
                )
                log(error_msg, 'ERROR')
                messagebox.showerror("Invalid Session", error_msg, parent=self.root)
                return False
            
            log(f"Loading session from: {session_dir}")
            
            # Initialize session manager
            self.session_manager = SessionManager(session_dir)
            
            # Load session state from file
            loaded_state = self.session_manager.load_session_from_file(session_dir)

            if loaded_state is None:
                messagebox.showerror("Error", "Failed to load session file", parent=self.root)
                return False
            
            # Try to load TIF from session folder
            tif_in_session = session_dir / Config.ORIGINAL_TIF_NAME
            
            if tif_in_session.exists():
                log(f"Loading TIF from session: {tif_in_session}")
                try:
                    self.image_handler = TIFHandler(tif_in_session)
                except Exception as e:
                    log(f"Error loading TIF from session: {e}", 'ERROR')
                    messagebox.showerror(
                        "Error",
                        f"Failed to load TIF from session:\n{str(e)}",
                        parent=self.root
                    )
                    return False
            else:
                # Fallback: ask user to locate TIF
                original_path = loaded_state.image_metadata.file_path
                
                response = messagebox.askyesno(
                    "TIF Not Found",
                    f"Original TIF not found in session folder.\n\n"
                    f"Original location:\n{original_path}\n\n"
                    f"Would you like to locate the TIF file manually?",
                    parent=self.root
                )
                
                if not response:
                    return False
                
                tif_path = filedialog.askopenfilename(
                    title="Locate Original TIF File",
                    filetypes=[("TIF files", "*.tif;*.tiff"), ("All files", "*.*")],
                    parent=self.root
                )
                
                if not tif_path:
                    return False
                
                try:
                    self.image_handler = TIFHandler(tif_path)
                except Exception as e:
                    log(f"Error loading TIF: {e}", 'ERROR')
                    messagebox.showerror(
                        "Error",
                        f"Failed to load TIF file:\n{str(e)}",
                        parent=self.root
                    )
                    return False
            

            # Set project info
            self.project_name = loaded_state.project_name
            self.project_dir = session_dir
            self.original_tif_path = tif_in_session if tif_in_session.exists() else Path(tif_path) if 'tif_path' in locals() else session_dir / Config.ORIGINAL_TIF_NAME
            
            # Restore session state
            self.restore_session_state(loaded_state)
            
            # Set project directory and output directory
            self.file_manager.current_project_dir = session_dir

            self.output_dir = session_dir.parent
            
            # Initialize propagation engine
            self.propagation_engine = PropagationEngine(self.image_handler)
            
            # Add to recent sessions
            # self.recent_sessions.add_session(session_dir, loaded_state)
            self.recent_sessions.add_session(session_dir, loaded_state)
            
            # Update display
            self.update_frame_info()
            self.update_display_now()
            
            messagebox.showinfo(
                "Session Loaded",
                f"Session loaded successfully!\n\n"
                f"Project: {loaded_state.project_name}\n"
                f"Frames: {loaded_state.image_metadata.total_frames}\n"
                f"Labeled: {len(loaded_state.labeled_frames)}\n"
                f"Channels: {len(loaded_state.channel_manager.get_all_channels())}",
                parent=self.root
            )
            
            log(f"Session loaded from {session_dir}")
            return True
            
        except Exception as e:
            log(f"Error loading session: {e}", 'ERROR')
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "Error",
                f"Failed to load session:\n{str(e)}",
                parent=self.root
            )
            return False
    

    def restore_session_state(self, session_state: SessionState):
        """Restore application state from loaded session - FIXED."""
        try:
            log("Restoring session state...")
            
            # Store session state
            self.session_state = session_state
            
            # If image_metadata is in session, verify it matches loaded TIF
            if session_state.image_metadata:
                if self.image_handler.metadata.total_frames != session_state.image_metadata.total_frames:
                    log(f"WARNING: TIF frame count mismatch! "
                        f"Expected {session_state.image_metadata.total_frames}, "
                        f"got {self.image_handler.metadata.total_frames}", 'WARNING')
            
                
            # CRITICAL FIX: Properly restore annotations
            self.frame_annotations = {}
            for frame_idx, annotation in session_state.annotations.items():
                self.frame_annotations[frame_idx] = annotation
            
            debug_log(f" DEBUG RESTORE: Copied {len(self.frame_annotations)} annotations to app")
            
            log(f"Restored {len(self.frame_annotations)} frame annotations")
            
            # Restore channel manager
            self.channel_manager = session_state.channel_manager
            channels = self.channel_manager.get_all_channels()
            
            debug_log(f" DEBUG RESTORE: Channel manager has {len(channels)} channels")
            
            if channels:
                # Restore active channel
                if hasattr(session_state, 'active_channel_id') and session_state.active_channel_id:
                    if session_state.active_channel_id in self.channel_manager.channels:
                        self.active_channel_id = session_state.active_channel_id
                        debug_log(f" DEBUG RESTORE: Restored active channel: {self.active_channel_id}")
                    else:
                        self.active_channel_id = channels[0].id
                        debug_log(f" DEBUG RESTORE: Active channel not found, using first: {self.active_channel_id}")
                else:
                    self.active_channel_id = channels[0].id
            
            # Update channel list UI
            self.collapsible_panel.update_channel_list()
            if channels:
                active_channel = self.channel_manager.get_channel(self.active_channel_id)
                if active_channel:
                    self.collapsible_panel._select_channel(active_channel)
            
            # Restore view state
            self.current_frame_idx = session_state.current_frame_index
            self.canvas.zoom_factor = session_state.zoom_factor
            self.canvas.pan_x = session_state.pan_x
            self.canvas.pan_y = session_state.pan_y
            
            # Restore tool state
            self.current_tool = session_state.current_tool
            self.brush_size = session_state.brush_size
            self.toolbar.brush_scale.set(self.brush_size)
            
            # Restore opacity
            if hasattr(session_state, 'mask_opacity'):
                self.mask_opacity = session_state.mask_opacity
                if hasattr(self, 'opacity_var'):
                    self.opacity_var.set(self.mask_opacity * 100)
                    if hasattr(self, 'opacity_label'):
                        self.opacity_label.config(text=f"{int(self.mask_opacity * 100)}%")
            
            # Restore enhancement settings
            self.enhancer.brightness = session_state.brightness
            self.enhancer.contrast = session_state.contrast
            self.enhancer.gaussian_blur = session_state.gaussian_blur
            self.enhancer.low_pass_filter = session_state.low_pass_filter
            self.enhancer.high_pass_filter = session_state.high_pass_filter
            self.enhancer.temporal_smoothing = session_state.temporal_smoothing
            
            # Update enhancement UI
            if hasattr(self, 'collapsible_panel'):
                self.collapsible_panel.brightness_var.set(session_state.brightness)
                self.collapsible_panel.contrast_var.set(session_state.contrast)
                self.collapsible_panel.gaussian_var.set(session_state.gaussian_blur)
                self.collapsible_panel.lowpass_var.set(session_state.low_pass_filter)
                self.collapsible_panel.highpass_var.set(session_state.high_pass_filter)
                self.collapsible_panel.temporal_var.set(session_state.temporal_smoothing)
            
            # Initialize frame_modified
            self.frame_modified = {}
            for frame_idx in range(self.image_handler.metadata.total_frames):
                self.frame_modified[frame_idx] = False
            
            # Clear undo stack
            self.undo_stack.clear()
            
            # Verify restoration
            labeled_count = len([a for a in self.frame_annotations.values() if a.has_annotations])
            log(f"Session restored: {labeled_count} labeled frames verified")
            
            debug_log(f" DEBUG RESTORE VERIFY: {labeled_count} frames have annotations")
            
            # Check first few annotations
            for idx in list(self.frame_annotations.keys())[:3]:
                ann = self.frame_annotations[idx]
                debug_log(f" DEBUG RESTORE VERIFY: Frame {idx} masks: {list(ann.channel_masks.keys())}")
            
            if labeled_count == 0 and len(session_state.labeled_frames) > 0:
                log(f"WARNING: Expected {len(session_state.labeled_frames)} frames but got {labeled_count}!", 'ERROR')
                messagebox.showwarning(
                    "Session Load Warning",
                    f"Session had {len(session_state.labeled_frames)} labeled frames\n"
                    f"but only {labeled_count} were restored.\n\n"
                    f"Some annotations may be missing!",
                    parent=self.root
                )
            
        except Exception as e:
            log(f"Error restoring session: {e}", 'ERROR')
            import traceback
            traceback.print_exc()


    def prev_frame(self):
        """Go to previous frame."""
        if not self.has_frames():
            return
        
        skip = self.navigation.skip_var.get()
        new_idx = max(0, self.current_frame_idx - skip)
        self.set_current_frame(new_idx)
    
    def next_frame(self):
        """Go to next frame."""
        if not self.has_frames():
            return
        
        skip = self.navigation.skip_var.get()
        max_idx = self.image_handler.metadata.total_frames - 1
        new_idx = min(max_idx, self.current_frame_idx + skip)
        self.set_current_frame(new_idx)
    
    def set_current_frame(self, frame_idx: int):
        
        """FIXED: Set current frame and update display."""
        if not self.has_frames():
            return
        
        max_idx = self.image_handler.metadata.total_frames - 1
        frame_idx = max(0, min(frame_idx, max_idx))
        
        if frame_idx == self.current_frame_idx:
            return
        
        self.current_frame_idx = frame_idx
        
        # Clear undo stack when changing frames
        self.undo_stack.clear()
        
        # Reset temporal smoothing
        self.enhancer.reset_temporal_smoothing()
        
        # FIXED: Clear propagation confidence label when changing frames manually
        if hasattr(self, 'confidence_label'):
            self.confidence_label.config(text="")
        
        # FIXED: Clear any active contour
        if hasattr(self.canvas, '_clear_contour_visual'):
            self.canvas._clear_contour_visual()
        
        self.update_frame_info()
        self.update_display_now()


    def update_frame_info(self):
        """Update frame counter and statistics."""
        if not self.has_frames():
            self.navigation.update_frame_label(0, 0)
            self.navigation.update_slider_position(0, 1)
            return
        
        total = self.image_handler.metadata.total_frames
        current = self.current_frame_idx
        
        self.navigation.update_frame_label(current, total)
        self.navigation.update_slider_position(current, total)
        
        # Update frame status indicator
        annotation = self.get_current_annotation()
        has_annotations = annotation is not None and annotation.has_annotations
        is_saved = not self.frame_modified.get(current, False)
        
        self.toolbar.update_frame_status(has_annotations, is_saved)
        
        # Update statistics if stats tab is visible
        if hasattr(self, 'collapsible_panel'):
            if not self.collapsible_panel.is_collapsed and self.collapsible_panel.current_tab == "Stats":
                self.collapsible_panel.update_statistics()
    
    def set_tool(self, tool: ToolType):
        """Set current drawing tool."""
        self.current_tool = tool
        self.toolbar.update_tool_indicator(tool)
        log(f"Tool: {tool.value}")
    
    def update_brush_size(self, value):
        """Update brush size from slider."""
        try:
            self.brush_size = float(value)
            self.toolbar.update_brush_value(self.brush_size)
        except Exception as e:
            debug_log(f"Error updating brush size: {e}")
    
    def update_opacity(self, value):
        """Update mask opacity."""
        try:
            # Convert from 0-100 UI scale to 0.0-1.0 internal scale
            self.mask_opacity = float(value) / 100.0
            self.opacity_label.config(text=f"{int(value)}%")
            self.schedule_display_update()
        except Exception as e:
            debug_log(f"Error updating opacity: {e}")
    
    def increase_opacity(self):
        """Increase mask opacity by 10%."""
        new_value = min(100, self.opacity_var.get() + 10)
        self.opacity_var.set(new_value)
        self.update_opacity(new_value)
    
    def decrease_opacity(self):
        """Decrease mask opacity by 10%."""
        new_value = max(0, self.opacity_var.get() - 10)
        self.opacity_var.set(new_value)
        self.update_opacity(new_value)
    
    def clear_mask(self):
        """Clear mask for current frame and active channel."""
        if not self.has_frames() or not self.active_channel_id:
            return
        
        response = messagebox.askyesno(
            "Clear Mask",
            "Clear mask for current channel on this frame?",
            parent=self.root
        )
        
        if response:
            annotation = self.get_current_annotation()
            if annotation:
                mask = self.image_handler.create_empty_mask(for_display=True)
                annotation.set_mask(self.active_channel_id, mask)
                
                # Mark frame as modified
                self.frame_modified[self.current_frame_idx] = True
                
                self.update_display_now()
                self.update_frame_info()
                log("Mask cleared")
    
    def clear_all_masks(self):
        """Clear ALL masks for ALL frames (destructive)."""
        if not self.has_frames():
            return
        
        labeled_count = len([a for a in self.frame_annotations.values() if a.has_annotations])
        
        response = messagebox.askyesno(
            "⚠️ Clear All Masks",
            f"This will PERMANENTLY DELETE all masks for all {labeled_count} labeled frames!\n\n"
            "This action CANNOT be undone!\n\n"
            "Are you absolutely sure?",
            icon='warning',
            parent=self.root
        )
        
        if not response:
            return
        
        # Double confirmation for safety
        confirm = messagebox.askyesno(
            "⚠️ Final Confirmation",
            "Last chance! Delete all annotations?",
            icon='warning',
            parent=self.root
        )
        
        if confirm:
            self.frame_annotations.clear()
            self.undo_stack.clear()
            self.frame_modified.clear() 
            self.frame_modified.clear()
            
            self.update_display_now()
            self.update_frame_info()
            
            messagebox.showinfo("Cleared", "All masks have been cleared", parent=self.root)
            log("All masks cleared")
    
    def undo_action(self):
        """Undo last drawing operation."""
        if not self.undo_stack.can_undo():
            return
        
        state = self.undo_stack.undo()
        if state:
            annotation = self.get_current_annotation()
            if annotation:
                annotation.set_mask(state.channel_id, state.mask_before.copy())
                self.update_display_now()
                log("Undo")
    
    def propagate_next(self):
        """Propagate masks forward to next frame."""
        if not self.has_frames() or not self.propagation_engine:
            return
        
        if self.current_frame_idx >= self.image_handler.metadata.total_frames - 1:
            messagebox.showinfo("Info", "Already at last frame", parent=self.root)
            return
        
        current_annotation = self.get_current_annotation()
        if not current_annotation or not current_annotation.has_annotations:
            messagebox.showwarning("Warning", "Current frame has no annotations to propagate", parent=self.root)
            return
        
        try:
            target_annotation, confidence = self.propagation_engine.propagate_forward(
                self.current_frame_idx,
                current_annotation,
                self.session_state.channel_manager
            )
            
            self.frame_annotations[target_annotation.frame_index] = target_annotation
            self.propagation_confidence = confidence
            
            # Mark target frame as modified
            self.frame_modified[target_annotation.frame_index] = True
            
            # Move to next frame
            self.set_current_frame(self.current_frame_idx + 1)
            
            # Update confidence display
            color = "#00FF00" if confidence > 80 else "#FFA500" if confidence > 60 else "#FF0000"
            self.confidence_label.config(
                text=f"Confidence: {confidence:.1f}%",
                foreground=color
            )
            
            log(f"Propagated forward (confidence: {confidence:.1f}%)")
            
        except Exception as e:
            log(f"Error propagating: {e}", 'ERROR')
            messagebox.showerror("Error", f"Propagation failed: {str(e)}", parent=self.root)
    
    def propagate_back(self):
        """Propagate masks backward to previous frame."""
        if not self.has_frames() or not self.propagation_engine:
            return
        
        if self.current_frame_idx <= 0:
            messagebox.showinfo("Info", "Already at first frame", parent=self.root)
            return
        
        current_annotation = self.get_current_annotation()
        if not current_annotation or not current_annotation.has_annotations:
            messagebox.showwarning("Warning", "Current frame has no annotations to propagate", parent=self.root)
            return
        
        try:
            target_annotation, confidence = self.propagation_engine.propagate_backward(
                self.current_frame_idx,
                current_annotation,
                self.session_state.channel_manager
            )
            
            self.frame_annotations[target_annotation.frame_index] = target_annotation
            self.propagation_confidence = confidence
            
            # Mark target frame as modified
            self.frame_modified[target_annotation.frame_index] = True
            
            # Move to previous frame
            self.set_current_frame(self.current_frame_idx - 1)
            
            # Update confidence display
            color = "#00FF00" if confidence > 80 else "#FFA500" if confidence > 60 else "#FF0000"
            self.confidence_label.config(
                text=f"Confidence: {confidence:.1f}%",
                foreground=color
            )
            
            log(f"Propagated backward (confidence: {confidence:.1f}%)")
            
        except Exception as e:
            log(f"Error propagating: {e}", 'ERROR')
            messagebox.showerror("Error", f"Propagation failed: {str(e)}", parent=self.root)
    
    def zoom_in(self):
        """Zoom in on canvas."""
        if self.has_frames():
            center_x = self.canvas.canvas.winfo_width() // 2
            center_y = self.canvas.canvas.winfo_height() // 2
            self.canvas._apply_zoom(True, center_x, center_y)
    
    def zoom_out(self):
        """Zoom out on canvas."""
        if self.has_frames():
            center_x = self.canvas.canvas.winfo_width() // 2
            center_y = self.canvas.canvas.winfo_height() // 2
            self.canvas._apply_zoom(False, center_x, center_y)
    
    def reset_view(self):
        """Reset zoom and pan."""
        self.canvas.reset_view()
    
    def schedule_display_update(self):
        """Schedule display update with throttling."""
        if not self.update_pending:
            self.update_pending = True
            self.root.after(Config.UPDATE_THROTTLE_MS, self.update_display_now)
    
    def update_display_now(self):
        """FIXED: Update display immediately with memory management."""
        self.update_pending = False
        
        if not self.has_frames():
            self.canvas.clear_canvas()
            return
        
        # OPTIMIZATION: During fast playback, throttle display updates
        if self.playback_controller.is_playing() and self.playback_controller.fps > 20:
            # For high FPS, only update every other frame
            if hasattr(self, '_last_display_frame'):
                if self.current_frame_idx - self._last_display_frame < 2:
                    return
            self._last_display_frame = self.current_frame_idx
        
        try:
            # Load and enhance frame
            frame = self.image_handler.load_frame(self.current_frame_idx, for_display=True)
            if frame is None:
                return
            
            frame = self.enhancer.apply_filters(frame)
            
            # Get canvas dimensions
            canvas_w = self.canvas.canvas.winfo_width()
            canvas_h = self.canvas.canvas.winfo_height()
            
            if canvas_w <= 1 or canvas_h <= 1:
                return
            
            # Calculate display dimensions with zoom
            display_w = int(self.image_handler.metadata.display_width * self.canvas.zoom_factor)
            display_h = int(self.image_handler.metadata.display_height * self.canvas.zoom_factor)
            
            # Resize frame if zoomed
            if self.canvas.zoom_factor != 1.0:
                if self.canvas.zoom_factor > 1.5:
                    interp = Config.ZOOM_RESIZE_METHOD
                else:
                    interp = Config.RESIZE_METHOD
                
                frame = cv2.resize(frame, (display_w, display_h), interpolation=interp)
            
            # Create composite with masks
            composite = frame.copy()
            
            annotation = self.get_current_annotation()
            if annotation:
                channel_manager = self.session_state.channel_manager
                
                # Overlay visible channels in reverse order (bottom to top)
                for channel in reversed(channel_manager.get_visible_channels()):
                    mask = annotation.get_mask(channel.id)
                    
                    if mask is not None and np.any(mask):
                        # Resize mask if zoomed
                        if self.canvas.zoom_factor != 1.0:
                            mask_display = cv2.resize(
                                mask,
                                (display_w, display_h),
                                interpolation=cv2.INTER_NEAREST
                            )
                        else:
                            mask_display = mask
                        
                        # Create colored overlay
                        overlay = composite.copy()
                        mask_pixels = np.where(mask_display > 0)
                        
                        if len(mask_pixels[0]) > 0:
                            overlay[mask_pixels] = channel.color_rgb
                            
                            # FIXED: Use self.mask_opacity (properly synchronized)
                            composite = cv2.addWeighted(
                                composite,
                                1.0 - self.mask_opacity,
                                overlay,
                                self.mask_opacity,
                                0
                            )
            
            # Calculate position with pan
            img_x = (canvas_w - display_w) // 2 + int(self.canvas.pan_x)
            img_y = (canvas_h - display_h) // 2 + int(self.canvas.pan_y)
            
            # FIXED: Always set display_info
            self.display_info = (img_x, img_y, display_w, display_h)
            
            # Convert to PhotoImage
            composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(composite_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.canvas.canvas.delete("all")
            self.canvas.canvas.create_image(img_x, img_y, anchor=tk.NW, image=photo)
            
            # FIXED: Keep reference to prevent garbage collection
            self.canvas.current_photo = photo
            self.canvas._photo_refs.append(photo)
            
            # FIXED: Limit photo references to prevent memory leak
            if len(self.canvas._photo_refs) > 5:
                self.canvas._photo_refs = self.canvas._photo_refs[-3:]
            
        except Exception as e:
            log(f"Error updating display: {e}", 'ERROR')
            debug_log(f"Display error traceback: {e}")



    def show_about(self):
        """Show about dialog."""
        about_text = f"""
{Config.APP_FULL_NAME} (MAT)
Version {VERSION}
Build: {BUILD_DATE}

Professional-grade annotation tool for large scientific images with:
- Dynamic multi-channel architecture
- Memory-efficient processing for 6K+ images  
- AI-powered mask propagation
- Session persistence with recent sessions
- Comprehensive image enhancement filters
- Self-contained sessions

Keyboard Shortcuts:
- D: Draw tool
- E: Erase tool
- A: Erase all channels
- C: Clear frame
- P: Propagate forward
- B: Propagate backward
- [/]: Decrease/Increase opacity
- +/-: Zoom in/out
- Arrows: Navigate frames
- Ctrl+S: Save current
- Ctrl+Z: Undo

Author: Development Team
License: MIT
"""
        messagebox.showinfo("About MAT", about_text, parent=self.root)
    
    def quit_application(self):
        """Quit application with cleanup."""
        if not messagebox.askyesno("Exit", "Are you sure you want to exit?", parent=self.root):
            return
        
        # Auto-save session if enabled
        if Config.AUTOSAVE_ON_EXIT and self.session_state and self.file_manager.current_project_dir:
            try:
                if self.session_manager:
                    log("Auto-saving session on exit...")
                    self.save_session()
            except Exception as e:
                log(f"Error auto-saving session: {e}", 'ERROR')
        
        # Cleanup
        if self.image_handler:
            self.image_handler.clear_cache()
        
        self.root.quit()
        self.root.destroy()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main application entry point."""
    try:
        log(f"Starting {Config.APP_NAME} v{VERSION}")
        
        # Create root window
        root = tk.Tk()
        root.title(f"{Config.APP_NAME} v{VERSION}")
        root.geometry(f"{Config.DEFAULT_WINDOW_WIDTH}x{Config.DEFAULT_WINDOW_HEIGHT}")
        root.minsize(Config.MIN_WINDOW_WIDTH, Config.MIN_WINDOW_HEIGHT)
        
        # Create application
        app = MATApplication(root)
        
        # Show welcome dialog
        welcome = WelcomeDialog(root, app.recent_sessions)
        result, selected_file, output_folder = welcome.show()
        
        if result == 'cancelled':
            root.destroy()
            return
        
        if result == 'file_selected' and selected_file:
            app.load_file(selected_file)
            if output_folder:
                app.output_dir = Path(output_folder)
        
        elif result == 'session_selected' and selected_file:
            session_path = Path(selected_file)
            app.load_session_with_fallback(session_path)
        
        # Start main loop
        root.mainloop()
        
    except KeyboardInterrupt:
        log("Application interrupted by user")
        
    except Exception as e:
        log(f"Fatal error: {e}", 'ERROR')
        import traceback
        traceback.print_exc()
        
        try:
            messagebox.showerror(
                "Fatal Error",
                f"Application encountered a fatal error:\n\n{str(e)}\n\n"
                "Please check the log for details."
            )
        except:
            pass
    
    finally:
        log(f"{Config.APP_NAME} shutdown")


if __name__ == "__main__":
    main()