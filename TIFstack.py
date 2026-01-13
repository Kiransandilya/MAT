#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration."""
    
    APP_NAME = "TIF Stack Manager"
    VERSION = "1.0.0"
    
    # UI Colors (Dark Theme)
    COLOR_BG_PRIMARY = '#2b2b2b'
    COLOR_BG_SECONDARY = '#333333'
    COLOR_FG_PRIMARY = '#ffffff'
    COLOR_FG_SECONDARY = '#cccccc'
    COLOR_ACCENT = '#4CAF50'
    COLOR_ERROR = '#F44336'
    
    # Window size
    WINDOW_WIDTH = 700
    WINDOW_HEIGHT = 1000
    
    # Defaults
    DEFAULT_CHUNK_SIZE = 200
    SUBFOLDER_SUFFIX = "_shortstacks"
    METADATA_FILENAME = "split_info.txt"


# =============================================================================
# TIF STACK PROCESSOR
# =============================================================================

class TIFStackProcessor:
    """Core logic for splitting and stitching TIF stacks."""
    
    @staticmethod
    def get_frame_count(tif_path: Path) -> int:
        """Get number of frames in TIF stack."""
        try:
            with Image.open(tif_path) as img:
                frame_count = 0
                try:
                    while True:
                        img.seek(frame_count)
                        frame_count += 1
                except EOFError:
                    pass
                return frame_count
        except Exception as e:
            raise ValueError(f"Cannot read TIF file: {e}")
    
    @staticmethod
    def split_stack(input_path: Path, output_dir: Path, chunk_size: int,
                   progress_callback=None) -> Tuple[Path, List[str]]:
        """
        Split TIF stack into smaller chunks.
        
        Args:
            input_path: Path to input TIF file
            output_dir: Base output directory
            chunk_size: Number of frames per chunk
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Tuple of (output_subfolder_path, list_of_created_files)
        """
        # Validate input
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Get total frames
        total_frames = TIFStackProcessor.get_frame_count(input_path)
        
        if total_frames == 0:
            raise ValueError("TIF file has no frames")
        
        # Create output subfolder
        base_name = input_path.stem
        output_subfolder = output_dir / f"{base_name}{Config.SUBFOLDER_SUFFIX}"
        output_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Calculate chunks
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        created_files = []
        
        # Process each chunk
        with Image.open(input_path) as img:
            for chunk_idx in range(num_chunks):
                start_frame = chunk_idx * chunk_size
                end_frame = min(start_frame + chunk_size, total_frames)
                actual_frames = end_frame - start_frame
                
                # Create output filename
                output_filename = f"{base_name}_Frames_{start_frame+1}-{end_frame}.tif"
                output_path = output_subfolder / output_filename
                
                # Extract frames for this chunk
                frames = []
                for frame_idx in range(start_frame, end_frame):
                    img.seek(frame_idx)
                    frames.append(img.copy())
                
                # Save chunk as multi-frame TIF
                if frames:
                    frames[0].save(
                        output_path,
                        save_all=True,
                        append_images=frames[1:] if len(frames) > 1 else [],
                        compression='tiff_lzw'
                    )
                    created_files.append(output_filename)
                
                # Progress callback
                if progress_callback:
                    progress_callback(
                        chunk_idx + 1,
                        num_chunks,
                        f"Creating chunk {chunk_idx + 1}/{num_chunks}: {output_filename}"
                    )
        
        # Create metadata file
        TIFStackProcessor._create_metadata_file(
            output_subfolder,
            input_path,
            total_frames,
            chunk_size,
            created_files
        )
        
        return output_subfolder, created_files
    
    @staticmethod
    def _create_metadata_file(output_dir: Path, original_file: Path,
                             total_frames: int, chunk_size: int,
                             created_files: List[str]):
        """Create metadata text file with split information."""
        metadata_path = output_dir / Config.METADATA_FILENAME
        
        with open(metadata_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TIF STACK SPLIT INFORMATION\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Original File: {original_file.name}\n")
            f.write(f"Original Path: {original_file}\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Chunk Size: {chunk_size} frames per file\n")
            f.write(f"Number of Chunks: {len(created_files)}\n")
            f.write(f"Split Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n{'=' * 70}\n")
            f.write("CREATED FILES:\n")
            f.write("=" * 70 + "\n\n")
            
            for idx, filename in enumerate(created_files, 1):
                # Parse frame range from filename
                if "_Frames_" in filename:
                    frame_range = filename.split("_Frames_")[1].replace(".tif", "")
                    f.write(f"{idx:3d}. {filename}\n")
                    f.write(f"     Frame Range: {frame_range}\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("STITCHING INSTRUCTIONS:\n")
            f.write("=" * 70 + "\n\n")
            f.write("To reconstruct the original TIF stack:\n")
            f.write(f"1. Keep all files in this folder: {output_dir.name}\n")
            f.write("2. Use the 'Stitch' function in TIF Stack Manager\n")
            f.write("3. Select this folder\n")
            f.write("4. The files will be automatically combined in the correct order\n")
    
    @staticmethod
    def stitch_stack(input_dir: Path, output_path: Path,
                    progress_callback=None) -> int:
        """
        Stitch TIF chunks back into single stack.
        
        Args:
            input_dir: Directory containing TIF chunks
            output_path: Path for output TIF file
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Total number of frames in stitched stack
        """
        # Find all TIF files matching pattern
        tif_files = sorted(
            [f for f in input_dir.glob("*_Frames_*.tif")],
            key=TIFStackProcessor._extract_start_frame
        )
        
        if not tif_files:
            raise ValueError("No TIF chunks found in directory")
        
        # Validate sequence
        TIFStackProcessor._validate_sequence(tif_files)
        
        # Collect all frames
        all_frames = []
        total_files = len(tif_files)
        
        for file_idx, tif_file in enumerate(tif_files):
            with Image.open(tif_file) as img:
                frame_count = 0
                try:
                    while True:
                        img.seek(frame_count)
                        all_frames.append(img.copy())
                        frame_count += 1
                except EOFError:
                    pass
            
            # Progress callback
            if progress_callback:
                progress_callback(
                    file_idx + 1,
                    total_files,
                    f"Loading chunk {file_idx + 1}/{total_files}: {tif_file.name}"
                )
        
        # Save combined stack
        if all_frames:
            if progress_callback:
                progress_callback(
                    total_files,
                    total_files,
                    f"Saving combined stack ({len(all_frames)} frames)..."
                )
            
            all_frames[0].save(
                output_path,
                save_all=True,
                append_images=all_frames[1:] if len(all_frames) > 1 else [],
                compression='tiff_lzw'
            )
        
        return len(all_frames)
    
    @staticmethod
    def _extract_start_frame(filename: Path) -> int:
        """Extract starting frame number from filename."""
        try:
            # Format: TIFFilename_Frames_1-200.tif
            parts = filename.stem.split("_Frames_")
            if len(parts) == 2:
                frame_range = parts[1]
                start_frame = int(frame_range.split("-")[0])
                return start_frame
        except:
            pass
        return 0
    
    @staticmethod
    def _validate_sequence(tif_files: List[Path]):
        """Validate that TIF chunks form a continuous sequence."""
        expected_start = 1
        
        for tif_file in tif_files:
            start_frame = TIFStackProcessor._extract_start_frame(tif_file)
            if start_frame != expected_start:
                raise ValueError(
                    f"Sequence gap detected!\n"
                    f"Expected frame {expected_start}, found {start_frame}\n"
                    f"File: {tif_file.name}"
                )
            
            # Extract end frame to get next expected start
            try:
                parts = tif_file.stem.split("_Frames_")[1]
                end_frame = int(parts.split("-")[1])
                expected_start = end_frame + 1
            except:
                raise ValueError(f"Cannot parse frame range: {tif_file.name}")


# =============================================================================
# GUI APPLICATION
# =============================================================================

class TIFStackManagerGUI:
    """Main GUI application."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.processor = TIFStackProcessor()
        
        # Setup window
        self.root.title(f"{Config.APP_NAME} v{Config.VERSION}")
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        self.root.configure(bg=Config.COLOR_BG_PRIMARY)
        
        # Apply theme
        self._setup_theme()
        
        # Create UI
        self._create_ui()
        
        # Center window
        self._center_window()
    
    def _setup_theme(self):
        """Apply dark theme."""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("TFrame", background=Config.COLOR_BG_PRIMARY)
        style.configure("TLabel", background=Config.COLOR_BG_PRIMARY, 
                       foreground=Config.COLOR_FG_PRIMARY, font=("Arial", 10))
        style.configure("Title.TLabel", font=("Arial", 14, "bold"))
        style.configure("TButton", font=("Arial", 10))
        style.configure("Accent.TButton", foreground=Config.COLOR_ACCENT)
        style.configure("TNotebook", background=Config.COLOR_BG_PRIMARY)
        style.configure("TNotebook.Tab", padding=[20, 10])
    
    def _center_window(self):
        """Center window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_ui(self):
        """Create main UI."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(
            main_frame,
            text=Config.APP_NAME,
            style="Title.TLabel"
        )
        title.pack(pady=(0, 20))
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Split tab
        split_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(split_frame, text="✂ Split Stack")
        self._create_split_tab(split_frame)
        
        # Stitch tab
        stitch_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(stitch_frame, text="🔗 Stitch Stack")
        self._create_stitch_tab(stitch_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 9)
        )
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def _create_split_tab(self, parent):
        """Create split stack tab."""
        # Input file section
        input_frame = ttk.LabelFrame(parent, text="Input", padding=15)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="TIF Stack File:").pack(anchor=tk.W)
        
        file_frame = ttk.Frame(input_frame)
        file_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.split_input_var = tk.StringVar()
        ttk.Entry(
            file_frame,
            textvariable=self.split_input_var,
            state='readonly',
            font=("Arial", 9)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(
            file_frame,
            text="Browse...",
            command=self._browse_split_input
        ).pack(side=tk.RIGHT)
        
        # Settings section
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding=15)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        chunk_frame = ttk.Frame(settings_frame)
        chunk_frame.pack(fill=tk.X)
        
        ttk.Label(chunk_frame, text="Frames per chunk:").pack(side=tk.LEFT)
        
        self.chunk_size_var = tk.IntVar(value=Config.DEFAULT_CHUNK_SIZE)
        chunk_spinbox = ttk.Spinbox(
            chunk_frame,
            from_=10,
            to=10000,
            textvariable=self.chunk_size_var,
            width=10,
            command=self._update_split_info  # ADD THIS LINE
        )
        chunk_spinbox.pack(side=tk.LEFT, padx=10)

        # Also bind to variable changes (for when user types directly)
        self.chunk_size_var.trace_add('write', lambda *args: self._update_split_info())
        
        
        # Output section
        output_frame = ttk.LabelFrame(parent, text="Output", padding=15)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(output_frame, text="Output Folder:").pack(anchor=tk.W)
        
        out_frame = ttk.Frame(output_frame)
        out_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.split_output_var = tk.StringVar()
        ttk.Entry(
            out_frame,
            textvariable=self.split_output_var,
            state='readonly',
            font=("Arial", 9)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(
            out_frame,
            text="Browse...",
            command=self._browse_split_output
        ).pack(side=tk.RIGHT)
        
        # Info section
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.split_info_text = tk.Text(
            info_frame,
            height=8,
            wrap=tk.WORD,
            font=("Courier", 9),
            bg=Config.COLOR_BG_SECONDARY,
            fg=Config.COLOR_FG_PRIMARY,
            state=tk.DISABLED
        )
        self.split_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.split_progress = ttk.Progressbar(
            parent,
            orient=tk.HORIZONTAL,
            mode='determinate'
        )
        self.split_progress.pack(fill=tk.X, pady=(0, 10))
        
        # Split button
        ttk.Button(
            parent,
            text="✂ Split Stack",
            command=self._run_split,
            style="Accent.TButton"
        ).pack(pady=10)
    
    def _create_stitch_tab(self, parent):
        """Create stitch stack tab."""
        # Input folder section
        input_frame = ttk.LabelFrame(parent, text="Input", padding=15)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            input_frame,
            text="Folder with Split TIFs:",
            font=("Arial", 10)
        ).pack(anchor=tk.W)
        
        ttk.Label(
            input_frame,
            text="(Select the *_shortstacks folder)",
            font=("Arial", 8),
            foreground=Config.COLOR_FG_SECONDARY
        ).pack(anchor=tk.W, pady=(0, 5))
        
        folder_frame = ttk.Frame(input_frame)
        folder_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.stitch_input_var = tk.StringVar()
        ttk.Entry(
            folder_frame,
            textvariable=self.stitch_input_var,
            state='readonly',
            font=("Arial", 9)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(
            folder_frame,
            text="Browse...",
            command=self._browse_stitch_input
        ).pack(side=tk.RIGHT)
        
        # Output section
        output_frame = ttk.LabelFrame(parent, text="Output", padding=15)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(output_frame, text="Save Combined Stack As:").pack(anchor=tk.W)
        
        out_frame = ttk.Frame(output_frame)
        out_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.stitch_output_var = tk.StringVar()
        ttk.Entry(
            out_frame,
            textvariable=self.stitch_output_var,
            state='readonly',
            font=("Arial", 9)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(
            out_frame,
            text="Browse...",
            command=self._browse_stitch_output
        ).pack(side=tk.RIGHT)
        
        # Info section
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.stitch_info_text = tk.Text(
            info_frame,
            height=8,
            wrap=tk.WORD,
            font=("Courier", 9),
            bg=Config.COLOR_BG_SECONDARY,
            fg=Config.COLOR_FG_PRIMARY,
            state=tk.DISABLED
        )
        self.stitch_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.stitch_progress = ttk.Progressbar(
            parent,
            orient=tk.HORIZONTAL,
            mode='determinate'
        )
        self.stitch_progress.pack(fill=tk.X, pady=(0, 10))
        
        # Stitch button
        ttk.Button(
            parent,
            text="🔗 Stitch Stack",
            command=self._run_stitch,
            style="Accent.TButton"
        ).pack(pady=10)
    
    # =========================================================================
    # SPLIT TAB METHODS
    # =========================================================================
    
    def _browse_split_input(self):
        """Browse for input TIF file."""
        filename = filedialog.askopenfilename(
            title="Select TIF Stack File",
            filetypes=[("TIF files", "*.tif;*.tiff"), ("All files", "*.*")]
        )
        
        if filename:
            self.split_input_var.set(filename)
            self._update_split_info()
    
    def _browse_split_output(self):
        """Browse for output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        
        if folder:
            self.split_output_var.set(folder)
    
    def _update_split_info(self):
        """Update split info display."""
        input_path = self.split_input_var.get()
        if not input_path:
            return
        
        try:
            input_file = Path(input_path)
            total_frames = self.processor.get_frame_count(input_file)
            chunk_size = self.chunk_size_var.get()
            num_chunks = (total_frames + chunk_size - 1) // chunk_size
            
            info = f"File: {input_file.name}\n"
            info += f"Total Frames: {total_frames:,}\n"
            info += f"Chunk Size: {chunk_size} frames\n"
            info += f"Number of Chunks: {num_chunks}\n\n"
            info += "Output files will be:\n"
            
            for i in range(min(num_chunks, 5)):
                start = i * chunk_size + 1
                end = min((i + 1) * chunk_size, total_frames)
                info += f"  • {input_file.stem}_Frames_{start}-{end}.tif\n"
            
            if num_chunks > 5:
                info += f"  ... and {num_chunks - 5} more\n"
            
            self._set_text(self.split_info_text, info)
            
        except Exception as e:
            self._set_text(self.split_info_text, f"Error: {str(e)}")
    
    def _run_split(self):
        """Execute split operation."""
        # Validate inputs
        input_path = self.split_input_var.get()
        output_dir = self.split_output_var.get()
        
        if not input_path:
            messagebox.showwarning("Missing Input", "Please select an input TIF file")
            return
        
        if not output_dir:
            messagebox.showwarning("Missing Output", "Please select an output folder")
            return
        
        try:
            chunk_size = self.chunk_size_var.get()
            
            if chunk_size < 1:
                messagebox.showwarning("Invalid Chunk Size", "Chunk size must be at least 1")
                return
            
            # Reset progress
            self.split_progress['value'] = 0
            self.split_progress['maximum'] = 100
            
            # Run split
            self.status_var.set("Splitting stack...")
            self.root.update()
            
            output_subfolder, created_files = self.processor.split_stack(
                Path(input_path),
                Path(output_dir),
                chunk_size,
                progress_callback=self._split_progress_callback
            )
            
            # Success
            self.status_var.set("Split complete!")
            messagebox.showinfo(
                "Success",
                f"Stack split successfully!\n\n"
                f"Created {len(created_files)} files in:\n"
                f"{output_subfolder}\n\n"
                f"See {Config.METADATA_FILENAME} for details."
            )
            
        except Exception as e:
            self.status_var.set("Error during split")
            messagebox.showerror("Error", f"Failed to split stack:\n\n{str(e)}")
    
    def _split_progress_callback(self, current: int, total: int, message: str):
        """Update split progress."""
        progress = int((current / total) * 100)
        self.split_progress['value'] = progress
        self.status_var.set(message)
        self.root.update()
    
    # =========================================================================
    # STITCH TAB METHODS
    # =========================================================================
    
    def _browse_stitch_input(self):
        """Browse for input folder with split TIFs."""
        folder = filedialog.askdirectory(
            title="Select Folder with Split TIFs (*_shortstacks folder)"
        )
        
        if folder:
            self.stitch_input_var.set(folder)
            self._update_stitch_info()
    
    def _browse_stitch_output(self):
        """Browse for output file."""
        filename = filedialog.asksaveasfilename(
            title="Save Combined Stack As",
            defaultextension=".tif",
            filetypes=[("TIF files", "*.tif"), ("All files", "*.*")]
        )
        
        if filename:
            self.stitch_output_var.set(filename)
    
    def _update_stitch_info(self):
        """Update stitch info display."""
        input_dir = self.stitch_input_var.get()
        if not input_dir:
            return
        
        try:
            input_path = Path(input_dir)
            
            # Find TIF chunks
            tif_files = sorted(
                [f for f in input_path.glob("*_Frames_*.tif")],
                key=self.processor._extract_start_frame
            )
            
            if not tif_files:
                self._set_text(self.stitch_info_text, "No TIF chunks found in folder")
                return
            
            # Calculate total frames
            total_frames = 0
            for tif_file in tif_files:
                total_frames += self.processor.get_frame_count(tif_file)
            
            info = f"Folder: {input_path.name}\n"
            info += f"Found {len(tif_files)} chunk(s)\n"
            info += f"Total Frames: {total_frames:,}\n\n"
            info += "Files to stitch:\n"
            
            for tif_file in tif_files[:10]:
                info += f"  • {tif_file.name}\n"
            
            if len(tif_files) > 10:
                info += f"  ... and {len(tif_files) - 10} more\n"
            
            self._set_text(self.stitch_info_text, info)
            
            # Auto-suggest output filename
            if not self.stitch_output_var.get():
                # Extract base name from first file
                base_name = tif_files[0].stem.split("_Frames_")[0]
                suggested_name = f"{base_name}_stitched.tif"
                suggested_path = input_path.parent / suggested_name
                self.stitch_output_var.set(str(suggested_path))
            
        except Exception as e:
            self._set_text(self.stitch_info_text, f"Error: {str(e)}")
    
    def _run_stitch(self):
        """Execute stitch operation."""
        # Validate inputs
        input_dir = self.stitch_input_var.get()
        output_file = self.stitch_output_var.get()
        
        if not input_dir:
            messagebox.showwarning("Missing Input", "Please select input folder")
            return
        
        if not output_file:
            messagebox.showwarning("Missing Output", "Please specify output file")
            return
        
        try:
            # Reset progress
            self.stitch_progress['value'] = 0
            self.stitch_progress['maximum'] = 100
            
            # Run stitch
            self.status_var.set("Stitching stack...")
            self.root.update()
            
            total_frames = self.processor.stitch_stack(
                Path(input_dir),
                Path(output_file),
                progress_callback=self._stitch_progress_callback
            )
            
            # Success
            self.status_var.set("Stitch complete!")
            messagebox.showinfo(
                "Success",
                f"Stack stitched successfully!\n\n"
                f"Total Frames: {total_frames:,}\n"
                f"Output: {Path(output_file).name}"
            )
            
        except Exception as e:
            self.status_var.set("Error during stitch")
            messagebox.showerror("Error", f"Failed to stitch stack:\n\n{str(e)}")
    
    def _stitch_progress_callback(self, current: int, total: int, message: str):
        """Update stitch progress."""
        progress = int((current / total) * 100)
        self.stitch_progress['value'] = progress
        self.status_var.set(message)
        self.root.update()
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _set_text(self, text_widget: tk.Text, content: str):
        """Set text in Text widget."""
        text_widget.config(state=tk.NORMAL)
        text_widget.delete('1.0', tk.END)
        text_widget.insert('1.0', content)
        text_widget.config(state=tk.DISABLED)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main application entry point."""
    try:
        root = tk.Tk()
        app = TIFStackManagerGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()