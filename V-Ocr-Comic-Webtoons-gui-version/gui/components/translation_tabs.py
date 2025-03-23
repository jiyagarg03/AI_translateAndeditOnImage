from PySide6.QtWidgets import QTabWidget, QWidget, QVBoxLayout
from PySide6.QtCore import Signal
from .image_preview import ImagePreview

class TranslationTabs(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)
        self.previews = {}  # Store ImagePreview instances
        
    def add_translation_tab(self, image_path: str):
        """Add a new tab with image preview"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        preview = ImagePreview(image_path)
        layout.addWidget(preview)
        
        # Get filename for tab title
        tab_name = image_path.split('/')[-1]
        self.addTab(tab, tab_name)
        self.previews[tab_name] = preview
        
    def get_current_preview(self) -> ImagePreview:
        """Get the ImagePreview widget of the current tab"""
        current_tab = self.currentWidget()
        if current_tab:
            return current_tab.layout().itemAt(0).widget()
        return None
    
    def close_tab(self, index: int):
        """Close the specified tab"""
        tab_name = self.tabText(index)
        if tab_name in self.previews:
            del self.previews[tab_name]
        self.removeTab(index) 