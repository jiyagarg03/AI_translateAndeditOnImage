from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QLabel, QTabWidget, QMessageBox)
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QIcon
import os

from pipeline import ComicTranslatePipeline
from .components.settings_panel import SettingsPanel
from .components.image_preview import ImagePreview
from .components.translation_tabs import TranslationTabs
from .components.drag_drop import DropArea

class TranslationWorker(QThread):
    finished = Signal(str)  # Emits path to translated image
    error = Signal(str)     # Emits error message
    progress = Signal(str)  # Emits progress messages
    
    def __init__(self, pipeline, image_path, source_lang, target_lang, context):
        super().__init__()
        self.pipeline = pipeline
        self.image_path = image_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.context = context
    
    def run(self):
        try:
            # Create output directory if it doesn't exist
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Get output path
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_translated.png")
            
            # Initialize inpainter pool
            self.progress.emit("Initializing inpainter...")
            self.pipeline.initialize_inpainter_pool(1)  # Use single inpainter for GUI
            
            # Run translation
            self.progress.emit("Starting translation process...")
            self.pipeline.process_image(
                self.image_path,
                self.source_lang,
                self.target_lang,
                self.context,
                show_masks=False  # Don't show masks in GUI mode
            )
            
            # Cleanup
            self.pipeline.inpainter_pool = None
            self.pipeline.free_inpainters = None
            
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Ensure cleanup happens even if there's an error
            if hasattr(self.pipeline, 'inpainter_pool') and self.pipeline.inpainter_pool:
                self.pipeline.inpainter_pool = None
                self.pipeline.free_inpainters = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comic Translator")
        self.setMinimumSize(QSize(1200, 800))
        
        # Initialize translation pipeline
        self.pipeline = ComicTranslatePipeline()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel for settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.settings_panel = SettingsPanel()
        self.drop_area = DropArea()
        
        left_layout.addWidget(self.settings_panel)
        left_layout.addWidget(self.drop_area)
        left_layout.addStretch()
        
        # Right panel for image preview and tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.translation_tabs = TranslationTabs()
        
        right_layout.addWidget(self.translation_tabs)
        
        # Add panels to main layout
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 3)
        
        # Connect signals
        self.drop_area.files_dropped.connect(self.handle_dropped_files)
        self.settings_panel.translate_requested.connect(self.start_translation)
        
        # Initialize dark mode
        self.set_dark_mode(True)
    
    def handle_dropped_files(self, file_paths):
        """Handle dropped image files"""
        for path in file_paths:
            self.translation_tabs.add_translation_tab(path)
    
    def start_translation(self, source_lang, target_lang, context):
        """Start the translation process"""
        preview = self.translation_tabs.get_current_preview()
        if not preview:
            QMessageBox.warning(self, "Warning", "Please select an image to translate first.")
            return
        
        # Show loading indicator
        preview.start_translation()
        
        # Create and start worker thread
        self.translation_worker = TranslationWorker(
            self.pipeline,
            preview.image_path,
            source_lang,
            target_lang,
            context
        )
        self.translation_worker.finished.connect(preview.translation_finished)
        self.translation_worker.error.connect(self.handle_translation_error)
        self.translation_worker.progress.connect(preview.update_status)
        self.translation_worker.start()
    
    def handle_translation_error(self, error_message):
        """Handle translation errors"""
        QMessageBox.critical(self, "Error", f"Translation failed: {error_message}")
        preview = self.translation_tabs.get_current_preview()
        if preview:
            preview.progress_bar.hide()
            preview.view_original_btn.setEnabled(True)
    
    def set_dark_mode(self, enabled: bool):
        """Toggle dark mode"""
        if enabled:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                QWidget {
                    background-color: #2d2d2d;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #0d47a1;
                    color: white;
                    border: none;
                    padding: 5px 15px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #1565c0;
                }
                QTabWidget::pane {
                    border: 1px solid #424242;
                }
                QTabBar::tab {
                    background: #424242;
                    color: #ffffff;
                    padding: 8px 20px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background: #0d47a1;
                }
                QProgressBar {
                    border: 1px solid #424242;
                    border-radius: 2px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #0d47a1;
                }
            """)
    