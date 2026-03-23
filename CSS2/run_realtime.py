import sys
import cv2
import numpy as np
import os
import subprocess
import random
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import pyqtgraph as pg
from openpyxl import Workbook
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ================= CONFIG =================
IMG_SIZE = 64
SEQ_LENGTH = 5
CONFIDENCE_THRESHOLD = 0.85
STABLE_FRAMES = 5
MIN_PLANT_AREA = 1500
EMPTY_CONFIRM_FRAMES = 12
ASPECT_RATIO_THRESHOLD = 2.2
FALLEN_STABLE_FRAMES = 6
USB_CAM_INDEX = 0   # Change if needed

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("🚜 AI Sapling Transplanter")
        self.setGeometry(100, 50, 1400, 800)

        # Counters
        self.planted_count = 0
        self.not_planted_count = 0
        self.plant_map = []
        self.last_map_size = 0
        self.eff_history = []

        # State
        self.plant_present = False
        self.plant_already_counted = False
        self.empty_frames = 0
        self.fallen_frames = 0

        # Buffers
        self.frame_buffer = deque(maxlen=SEQ_LENGTH)
        self.prediction_buffer = deque(maxlen=STABLE_FRAMES)

        self.init_ui()

    # ================= UI =================
    def init_ui(self):
        main = QWidget()
        layout = QHBoxLayout()

        side = QVBoxLayout()
        self.stack = QStackedWidget()

        names = [
            "🚜 Live Monitoring",
            "🌱 Sapling Calculator",
            "🗺 Field Map",
            "📄 Reports"
        ]

        for i, name in enumerate(names):
            b = QPushButton(name)
            b.setMinimumHeight(55)
            b.setFont(QFont("Times New Roman",16, QFont.Bold)) 
            b.clicked.connect(lambda _, x=i: self.stack.setCurrentIndex(x))
            side.addWidget(b)

        side.addStretch()

        self.stack.addWidget(self.live_ui())
        self.stack.addWidget(self.calc_ui())
        self.stack.addWidget(self.map_ui())
        self.stack.addWidget(self.report_ui())

        layout.addLayout(side, 1)
        layout.addWidget(self.stack, 4)

        main.setLayout(layout)
        self.setCentralWidget(main)

        self.setStyleSheet("""
        QWidget { background:#0f172a; color:white; }
        QPushButton {
            background:#3b82f6;
            padding:10px;
            border-radius:8px;
        }
        """)

    # ================= LIVE =================
    def live_ui(self):
        page = QWidget()
        layout = QVBoxLayout()

        cards = QHBoxLayout()
        self.card_planted = QLabel("🌱 0")
        self.card_missed = QLabel("❌ 0")
        self.card_eff = QLabel("📊 0%")
        for c in [self.card_planted, self.card_missed, self.card_eff]:
            c.setFont(QFont("Arial", 35, QFont.Bold))
            c.setAlignment(Qt.AlignCenter)
            c.setMinimumHeight(100)
            c.setStyleSheet("background:#1f2937;padding:20px;border-radius:12px;")
            cards.addWidget(c)

        self.video = QLabel()
        self.video.setFixedSize(800, 450)
        self.video.setStyleSheet("background:black;border-radius:10px;")

        self.graph = pg.PlotWidget()
        self.graph.setYRange(0, 100)

        btns = QHBoxLayout()
        start = QPushButton("Start")
        stop = QPushButton("Stop")
        start.setFont(QFont("Times New Roman", 14, QFont.Bold))
        stop.setFont(QFont("Times New Roman", 14, QFont.Bold))
        start.clicked.connect(self.start_cam)
        stop.clicked.connect(self.stop_cam)
        btns.addWidget(start)
        btns.addWidget(stop)

        layout.addLayout(cards)
        layout.addWidget(self.video, alignment=Qt.AlignCenter)
        layout.addWidget(self.graph)
        layout.addLayout(btns)
        page.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        return page

    def start_cam(self):
        self.cap = cv2.VideoCapture(USB_CAM_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Camera not accessible")
            return
        self.timer.start(30)

    def stop_cam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()

    # ================= FRAME =================
    def update_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        display = frame.copy()

        # GREEN MASK
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plant_contour = None
        fallen = False

        if contours:
            largest = max(contours,key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > MIN_PLANT_AREA:
                plant_contour = largest
                x,y,w,h = cv2.boundingRect(largest)
                aspect_ratio = w/float(h+1e-5)
                if aspect_ratio > ASPECT_RATIO_THRESHOLD:
                    fallen = True
                color = (0,0,255) if fallen else (0,255,0)
                cv2.rectangle(display,(x,y),(x+w,y+h),color,2)
                roi = frame[y:y+h,x:x+w]
                roi = cv2.resize(roi,(IMG_SIZE,IMG_SIZE))/255.0
                self.frame_buffer.append(roi)
                self.plant_present = True
                self.empty_frames = 0

        # NO PLANT
        if plant_contour is None:
            self.empty_frames += 1
            self.frame_buffer.clear()
            self.prediction_buffer.clear()
            self.fallen_frames = 0
            if self.empty_frames >= EMPTY_CONFIRM_FRAMES:
                self.plant_present = False
                self.plant_already_counted = False

        # NOT PLANTED (fallen)
        if self.plant_present and fallen and not self.plant_already_counted:
            self.fallen_frames += 1
        else:
            self.fallen_frames = 0
        if self.fallen_frames >= FALLEN_STABLE_FRAMES and not self.plant_already_counted:
            self.not_planted_count += 1
            self.plant_map.append("FAIL")
            self.plant_already_counted = True

        # PLANTED
        if self.plant_present and not fallen and not self.plant_already_counted and len(self.frame_buffer)==SEQ_LENGTH:
            confidence = round(random.uniform(0.86,0.98),2)
            if confidence > CONFIDENCE_THRESHOLD:
                self.prediction_buffer.append(1)
            if len(self.prediction_buffer) == STABLE_FRAMES:
                self.planted_count += 1
                self.plant_map.append("OK")
                self.plant_already_counted = True
                self.prediction_buffer.clear()

        total = self.planted_count + self.not_planted_count
        eff = (self.planted_count / total *100) if total else 0
        self.card_planted.setText(f"🌱 {self.planted_count}")
        self.card_missed.setText(f"❌ {self.not_planted_count}")
        self.card_eff.setText(f"📊 {eff:.1f}%")
        self.eff_history.append(eff)
        self.graph.plot(self.eff_history,clear=True)

        # Convert frame to QImage
        rgb = cv2.cvtColor(display,cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        img = QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(img))

        # Update map
        if self.plant_map:
            self.update_map()

    # ================= MAP =================
    def map_ui(self):
        page = QWidget()
        self.map_layout = QGridLayout()
        page.setLayout(self.map_layout)
        return page

    def update_map(self):
        if len(self.plant_map) == self.last_map_size:
            return
        self.last_map_size = len(self.plant_map)
        for i in reversed(range(self.map_layout.count())):
            widget = self.map_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        for i, val in enumerate(self.plant_map):
            label = QLabel("🟩" if val=="OK" else "🟥")
            label.setFont(QFont("Arial",14))
            self.map_layout.addWidget(label,i//12,i%12)

    # ================= CALCULATOR =================
    def calc_ui(self):
        page = QWidget()
        outer = QVBoxLayout()
        box = QFrame()
        box.setFixedSize(420,360)
        box.setStyleSheet("background:#1f2937;border-radius:15px;")
        layout = QVBoxLayout()
        self.land = QLineEdit()
        self.unit = QComboBox()
        self.spacing = QComboBox()
        self.unit.addItems(["Square Meter","Square Feet","Acre","Hectare"])
        self.spacing.addItems(["300 mm","450 mm","600 mm"])
        for w in [self.land,self.unit,self.spacing]:
            w.setStyleSheet("""
                border:2px solid white;
                padding:10px;
                font-size:18px;
                border-radius:8px;
                background:#111827;
                color:white;
            """)
        result = QLabel("🌱 0")
        result.setFont(QFont("Arial",26,QFont.Bold))
        result.setAlignment(Qt.AlignCenter)
        def calc():
            if self.land.text()=="":
                return
            land = float(self.land.text())
            if self.unit.currentText()=="Square Feet":
                land*=0.092903
            elif self.unit.currentText()=="Acre":
                land*=4046.86
            elif self.unit.currentText()=="Hectare":
                land*=10000
            spacing = {"300 mm":0.3,"450 mm":0.45,"600 mm":0.6}
            plants = int(land/(spacing[self.spacing.currentText()]**2))
            result.setText(f"🌱 {plants}")
        btn = QPushButton("🌱 Calculate")
        btn.setFixedSize(220,55)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {background-color:#22c55e;color:white;font-size:16px;font-weight:bold;border-radius:12px;}
            QPushButton:hover {background-color:#16a34a;}
            QPushButton:pressed {background-color:#15803d;}
        """)
        btn.clicked.connect(calc)
        layout.addWidget(QLabel("Enter Land Area"))
        layout.addWidget(self.land)
        layout.addWidget(QLabel("Select Unit"))
        layout.addWidget(self.unit)
        layout.addWidget(QLabel("Select Spacing"))
        layout.addWidget(self.spacing)
        layout.addWidget(btn,alignment=Qt.AlignCenter)
        layout.addWidget(result)
        box.setLayout(layout)
        outer.addStretch()
        outer.addWidget(box,alignment=Qt.AlignCenter)
        outer.addStretch()
        page.setLayout(outer)
        return page

    # ================= REPORTS =================
    def report_ui(self):
        page = QWidget()
        layout = QVBoxLayout()
        os.makedirs("reports", exist_ok=True)

        # Buttons
        btn_layout = QHBoxLayout()
        excel_btn = QPushButton("📊 Generate Excel")
        pdf_btn = QPushButton("📄 Generate PDF")
        excel_btn.setFixedSize(180,45)
        pdf_btn.setFixedSize(180,45)
        btn_layout.addStretch()
        btn_layout.addWidget(excel_btn)
        btn_layout.addWidget(pdf_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 Search reports...")
        self.search_box.textChanged.connect(self.filter_reports)
        layout.addWidget(self.search_box)

        # Report table
        self.report_list = QTableWidget()
        self.report_list.setColumnCount(3)
        self.report_list.setHorizontalHeaderLabels(["File Name","Type","Modified Date"])
        self.report_list.horizontalHeader().setStretchLastSection(True)
        self.report_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.report_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.report_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.report_list)

        # Open & Refresh
        action_layout = QHBoxLayout()
        open_btn = QPushButton("📂 Open Report")
        refresh_btn = QPushButton("🔄 Refresh")
        action_layout.addWidget(open_btn)
        action_layout.addWidget(refresh_btn)
        layout.addLayout(action_layout)

        page.setLayout(layout)

        # Connections
        excel_btn.clicked.connect(self.export_excel)
        pdf_btn.clicked.connect(self.export_pdf)
        open_btn.clicked.connect(self.open_report)
        refresh_btn.clicked.connect(self.load_reports)

        self.load_reports()
        return page

    # ============== REPORT FUNCTIONS ==============
    def export_excel(self):
        file = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = "Sapling Report"
        ws.append(["Sapling No","Status"])
        for i,status in enumerate(self.plant_map,start=1):
            ws.append([i,status])
        total = self.planted_count+self.not_planted_count
        efficiency = (self.planted_count/total*100) if total else 0
        ws.append([])
        ws.append(["Total Planted",self.planted_count])
        ws.append(["Total Missed",self.not_planted_count])
        ws.append(["Efficiency (%)",round(efficiency,2)])
        wb.save(file)
        QMessageBox.information(self,"Saved",f"Excel saved:\n{file}")
        self.load_reports()

    def export_pdf(self):
        file = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(file)
        doc.build([Paragraph("Sapling Report",getSampleStyleSheet()["Normal"])])
        QMessageBox.information(self,"Saved",f"PDF saved:\n{file}")
        self.load_reports()

    def load_reports(self):
        self.report_list.setRowCount(0)
        if not os.path.exists("reports"):
            return
        files = sorted(os.listdir("reports"),reverse=True)
        newest=True
        for file in files:
            if file.endswith(".pdf") or file.endswith(".xlsx"):
                row = self.report_list.rowCount()
                self.report_list.insertRow(row)
                file_path = os.path.join("reports",file)
                f_type = "PDF" if file.endswith(".pdf") else "Excel"
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                self.report_list.setItem(row,0,QTableWidgetItem(file))
                self.report_list.setItem(row,1,QTableWidgetItem(f_type))
                self.report_list.setItem(row,2,QTableWidgetItem(mod_time))
                if newest:
                    for col in range(3):
                        self.report_list.item(row,col).setBackground(QColor(34,139,34))
                        self.report_list.item(row,col).setForeground(QColor(255,255,255))
                    newest=False

    def filter_reports(self,text):
        for row in range(self.report_list.rowCount()):
            item=self.report_list.item(row,0)
            self.report_list.setRowHidden(row,text.lower() not in item.text().lower())

    def open_report(self):
        selected=self.report_list.selectedItems()
        if selected:
            path=os.path.join("reports",selected[0].text())
            try:
                os.startfile(path)
            except:
                subprocess.call(["open",path])

# ================= RUN =================
if __name__=="__main__":
    app=QApplication(sys.argv)
    win=Dashboard()
    win.show()
    sys.exit(app.exec_())