from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                              QHBoxLayout, QProgressBar)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal

class ImagePreview(QWidget):
    translation_complete = Signal()
    
    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path
        layout = QVBoxLayout(self)
        
        # Preview controls
        controls = QHBoxLayout()
        self.view_original_btn = QPushButton("Original")
        self.view_translated_btn = QPushButton("Translated")
        controls.addWidget(self.view_original_btn)
        controls.addWidget(self.view_translated_btn)
        controls.addStretch()
        layout.addLayout(controls)
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.hide()
        layout.addWidget(self.status_label)
        
        # Loading indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Load and display image
        self.original_pixmap = QPixmap(image_path)
        self.translated_pixmap = None
        self.show_original()
        
        # Connect buttons
        self.view_original_btn.clicked.connect(self.show_original)
        self.view_translated_btn.clicked.connect(self.show_translated)
        
        # Initially disable translated button
        self.view_translated_btn.setEnabled(False)
    
    def show_original(self):
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
    
    def show_translated(self):
        if self.translated_pixmap:
            scaled_pixmap = self.translated_pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
    
    def start_translation(self):
        """Show loading indicator when translation starts"""
        self.progress_bar.setRange(0, 0)  # Infinite progress
        self.progress_bar.show()
        self.status_label.show()
        self.status_label.setText("Initializing translation...")
        self.view_original_btn.setEnabled(False)
        self.view_translated_btn.setEnabled(False)
    
    def update_status(self, message: str):
        """Update status message"""
        self.status_label.setText(message)
    
    def translation_finished(self, translated_image_path: str):
        """Update with translated image and hide loading indicator"""
        self.progress_bar.hide()
        self.status_label.hide()
        self.view_original_btn.setEnabled(True)
        self.view_translated_btn.setEnabled(True)
        self.translated_pixmap = QPixmap(translated_image_path)
        self.show_translated()
        self.translation_complete.emit() 