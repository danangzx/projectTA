from kivymd.app import MDApp
from kivy.lang import Builder
from plyer import filechooser
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout

kv = """
MDBoxLayout:
    orientation:"vertical"
    spacing:20
    padding:20
    Image:
        id: img
    MDRaisedButton:
        text: "upload"
        on_release:
            app.file_chooser()
    MDRaisedButton:
        text: "proses"
        on_release:
            app.detection()
    
    
"""


class CounterApp(MDApp):    

    def build(self):
        return Builder.load_string(kv)

    def file_chooser(self):
        filechooser.open_file(on_selection=self.selected)

    def selected(self,selection):
        image_source = selection[0]
        if selection:
            self.root.ids.img.source = image_source
        print(image_source)
    
    def detection(self):
        filechooser.open_file(on_selection=self.selected)
        
        

if __name__ == '__main__':
    CounterApp().run()
