from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from plyer import filechooser

import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

class DetectionApp(MDApp):
    def build(self):
        self.layout = MDBoxLayout(
            orientation="vertical",
            spacing=20,
            padding=20
        )
        self.upload_button = MDRaisedButton(
            text="Upload",
            on_release=self.upload_image
        )
        self.press_button = MDRaisedButton(
            text="Proses",
            on_release=self.load_image
        )
        self.image = Image()
        self.layout.add_widget(self.image)
        self.layout.add_widget(self.upload_button)
        self.layout.add_widget(self.press_button)
        return self.layout

    def upload_image(self,instance):
        filechooser.open_file(on_selection=self.selected)
    
    def selected(self,selection):
        self.image_upload = selection[0]
        image_source = selection[0]
        if selection:
            self.image.source = image_source

    def load_image(self,instance):
        self.road_damage_metadata = MetadataCatalog.get("road_damage")
        self.road_damage_metadata.thing_classes = ["Road-damage", "alligator cracking","lateral cracking","longitudinal cracking","pothole"]

        self.cfg = get_cfg()

        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

        self.cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        self.cfg.SOLVER.STEPS = []        # do not decay learning rate
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        # self.cfg.MODEL.WEIGHTS = "model_final.pth"
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

        # Baca gambar menggunakan OpenCV
        image = cv2.imread(self.image_upload)  # Ganti dengan lokasi gambar Anda
        predictions = self.predictor(image)

        viz = Visualizer(image[:, :, ::-1], metadata = self.road_damage_metadata, )
        
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        img_output = output.get_image()[:,:,::-1]

        # Pastikan gambar berhasil dimuat
        if img_output is not None:
            # Konversi dari BGR ke RGB
            image_rgb = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

            # Buat objek Texture dari gambar
            texture = Texture.create(size=(img_output.shape[1], img_output.shape[0]))
            texture.flip_vertical()
            texture.blit_buffer(image_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

            # Tampilkan gambar di Kivy
            self.image.texture = texture
        else:
            print("Gagal memuat gambar.")

if __name__ == '__main__':
    app = DetectionApp()
    app.run()
