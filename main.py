from kivy.lang import Builder

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

KV = '''
BoxLayout:
    orientation: 'vertical'

    MDFloatLayout:
        Image:
            id: img
            pos_hint: {"center_x": .5, "center_y": .52}
            

    MDBoxLayout:
        orientation: 'horizontal'
        padding: 5
        spacing: 5
        size_hint: 1, 0.12
        MDRaisedButton:
            text: "Upload"
            size_hint: None, None
            size: "150dp", "40dp"
            on_release:
                app.upload_image()

        MDRaisedButton:
            text: "Detect"
            size_hint: None, None
            size: "150dp", "40dp"
            on_release:
                app.load_image()

        MDLabel:
            id: value_label
            text: ""
            size_hint_y: None
            height: "40dp"
'''

class DetectionApp(MDApp):

    def build(self):
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
        return Builder.load_string(KV)
    


    def nilai_persen(self,nilai):
        nilai_dibulatkan = round(nilai, 2)
        nilai_persen = nilai_dibulatkan * 100
        teks_nilai = f"{int(nilai_persen)}"
        return teks_nilai
    
    def ubah_ke_integer(self,arr):
        return [[int(x) for x in row] for row in arr]
    
    def hitung_objek_asli(self,koordinat_deteksi):

        x1,y1,x2,y2 = koordinat_deteksi
        
        luas = np.sqrt((x2-x1)**2 * (y2-y1)**2)
        
        return luas
    
    def get_2d_array_string(self,arr1):
        formatted_text = ""
        jumlah_lubang = 0
        jumlah_retak_memanjang = 0
        jumlah_retak_melintang = 0
        jumlah_retak_buaya = 0
        luas_lubang = 0
        luas_retak_memanjang = 0
        luas_retak_melintang = 0
        luas_retak_buaya = 0
        
        
        #menghitung kelas yang terdeteksi
        for item in arr1:
            jenis = item[0]
            nilai = int(item[1])
            if jenis == 'lubang':
                jumlah_lubang += 1
                luas_lubang += nilai
            elif jenis == 'retak memanjang':
                jumlah_retak_memanjang += 1
                luas_retak_memanjang += nilai
            elif jenis == 'retak melintang':
                jumlah_retak_melintang += 1
                luas_retak_melintang += nilai
            elif jenis == 'retak buaya':
                jumlah_retak_buaya += 1
                luas_retak_buaya += nilai
        
        #
        jumlah_array = []
        if jumlah_lubang > 0:
            jumlah_array.append((f"Jumlah lubang = {jumlah_lubang},                         total luas lubang ={luas_lubang}px\n"))
        if jumlah_retak_memanjang > 0:
            jumlah_array.append((f"Jumlah retak memanjang = {jumlah_retak_memanjang},       total luas retak memanjang ={luas_retak_memanjang}px\n"))
        if jumlah_retak_melintang > 0:
            jumlah_array.append((f"Jumlah retak melintang = {jumlah_retak_melintang},          total luas melintang ={luas_retak_melintang}px\n"))
        if jumlah_retak_buaya > 0:
            jumlah_array.append((f"Jumlah retak buaya = {jumlah_retak_buaya},                 total luas retak buaya ={luas_retak_buaya}px\n"))
        
        #menampilkan pada layar
        for item in jumlah_array:
            formatted_text += item            
            
        self.root.ids.value_label.text = formatted_text

    def buat_kotak(self,koordinat_int, total_ukur,teks_foto):        
        x1,y1,x2,y2 = koordinat_int

        self.tinggi, self.lebar, _ = self.foto.shape
        

        if self.tinggi <=800 or self.lebar <= 800 :

            # Gambar kotak pada objek
            # cv2.rectangle(foto, (x1,y1), (x2,y2), (0, 255, 0), 2)
            if( total_ukur <= 1000 ):
                #hijau
                cv2.rectangle(self.foto, (x1,y1), (x2,y2), (0, 255, 0), 2)
                cv2.putText(self.foto, teks_foto, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif (total_ukur <= 5000):
                #kuning
                cv2.rectangle(self.foto, (x1,y1), (x2,y2), (0, 255, 255), 2)
                cv2.putText(self.foto, teks_foto, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                #merah
                cv2.rectangle(self.foto, (x1,y1), (x2,y2), (0, 0, 255), 2)
                cv2.putText(self.foto, teks_foto, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)            

        else:

            if( total_ukur <= 50000 ):
                #hijau
                cv2.rectangle(self.foto, (x1,y1), (x2,y2), (0, 255, 0), 10)
                cv2.putText(self.foto, teks_foto, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            elif (total_ukur <= 500000):
                #kuning
                cv2.rectangle(self.foto, (x1,y1), (x2,y2), (0, 255, 255), 10)
                cv2.putText(self.foto, teks_foto, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
            else:
                #merah
                cv2.rectangle(self.foto, (x1,y1), (x2,y2), (0, 0, 255), 10)
                cv2.putText(self.foto, teks_foto, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                

    
    def load_image(self):

        #nilai array teks yang akan ditampilkan
        teks = []

        #nilai skor dari prediksi menjadi array
        nilai_skor = []

        #variabel hasil ukuran
        hasil = []

        # Baca gambar menggunakan OpenCV
        image = cv2.imread(self.image_upload)  # Ganti dengan lokasi gambar Anda
        self.foto = image
        predictions = self.predictor(image)

        # Mengubah nilai float menjadi integer
        #ubah pred_box ke list
        pred_box = predictions["instances"].pred_boxes.tensor.tolist()
        pred_box_int = self.ubah_ke_integer(pred_box)

        #nilai kelas menjadi array
        class_box = predictions["instances"].pred_classes.tolist()
        for i in range(len(class_box)):
            if class_box[i] == 1:
                class_box[i] = "retak buaya"
            if class_box[i] == 2:
                class_box[i] = "retak melintang"
            if class_box[i] == 3:
                class_box[i] = "retak memanjang"
            if class_box[i] == 4:
                class_box[i] = "lubang"

        #nilai skor dari prediksi menjadi array
        scores_box = predictions["instances"].scores.tolist()

        for i in range(len(scores_box)):
            nilai_skor.append(self.nilai_persen(scores_box[i]))

        for i in range(len(pred_box)):
            hasil.append(self.hitung_objek_asli(pred_box[i]))
        
        for i in range(len(class_box)):
            # Menentukan teks yang akan ditampilkan
            teks.append(f"{class_box[i]}= {nilai_skor[i]}%")

        # Menggabungkan dua array menjadi array 2 dimensi
        combined_2d_array = [[class_box[i], hasil[i]] for i in range(min(len(class_box), len(hasil)))]

        self.get_2d_array_string(combined_2d_array)

        # mencetak hasil
        for j in range(len(pred_box_int)):
            self.buat_kotak(pred_box_int[j],hasil[j],teks[j])

        # Pastikan gambar berhasil dimuat
        # image
        if image is not None:
            # Konversi dari BGR ke RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Buat objek Texture dari gambar
            texture = Texture.create(size=(image.shape[1], image.shape[0]))
            texture.flip_vertical()
            texture.blit_buffer(image_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

            # Tampilkan gambar di Kivy
            self.root.ids.img.texture = texture
        else:
            print("Gagal memuat gambar.")

    def upload_image(self):
        filechooser.open_file(on_selection=self.selected)
    
    def selected(self,selection):
        self.image_upload = selection[0]
        image_source = selection[0]
        if selection:
            self.root.ids.img.source = image_source


if __name__ == '__main__':
    DetectionApp().run()
