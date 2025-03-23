from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent

class DropArea(QLabel):
    files_dropped = Signal(list)  # Signal emitted when files are dropped
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 8px;
                padding: 20px;
            }
            QLabel:hover {
                border-color: #0d47a1;
            }
        """)
        self.setText("Drop images here\nor click to browse")
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(file_path)
        
        if files:
            self.files_dropped.emit(files) 