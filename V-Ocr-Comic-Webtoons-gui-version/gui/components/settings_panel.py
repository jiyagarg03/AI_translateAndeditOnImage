from PySide6.QtWidgets import (QWidget, QVBoxLayout, QComboBox, 
                              QLabel, QTextEdit, QPushButton)
from PySide6.QtCore import Signal

class SettingsPanel(QWidget):
    translate_requested = Signal(str, str, str)  # source_lang, target_lang, context
    
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Source Language
        layout.addWidget(QLabel("Source Language:"))
        self.source_lang = QComboBox()
        self.source_lang.addItems(["Japanese", "Korean", "Chinese", "English"])
        layout.addWidget(self.source_lang)
        
        # Target Language
        layout.addWidget(QLabel("Target Language:"))
        self.target_lang = QComboBox()
        self.target_lang.addItems(["English", "Spanish", "French", "German"])
        layout.addWidget(self.target_lang)
        
        # Extra Context
        layout.addWidget(QLabel("Extra Context:"))
        self.context = QTextEdit()
        self.context.setPlaceholderText("Enter any additional context or translation hints...")
        self.context.setMaximumHeight(100)
        layout.addWidget(self.context)
        
        # Translate Button
        self.translate_btn = QPushButton("Translate")
        self.translate_btn.clicked.connect(self._on_translate_clicked)
        layout.addWidget(self.translate_btn)
        
        layout.addStretch()
    
    def _on_translate_clicked(self):
        self.translate_requested.emit(
            self.source_lang.currentText(),
            self.target_lang.currentText(),
            self.context.toPlainText()
        )