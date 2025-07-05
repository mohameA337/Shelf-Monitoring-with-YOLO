# app.py

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from utils.data_classes import Shelf, Product

product_model = YOLO('best_pro.pt')
shelf_model = YOLO('best_shelves.pt')  

def detect_shelves(img):
    """
    Detect shelves using an OBB-trained YOLO model.
    Returns Shelf objects with oriented box coordinates.
    """
    results = shelf_model(img, imgsz=128, conf=0.1)
    shelves = []
    if results[0].obb is not None:
        for box in results[0].obb.xyxyxyxy:
            shelves.append(Shelf(box))
    return shelves

def detect_products(img):
    """
    Detect products using standard YOLO bbox model.
    """
    results = product_model(img)
    products = []
    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            products.append(Product(box))
    return products

def draw_boxes(img, objects, color=(0,255,0), thickness=5):
    """
    Draw bounding boxes or oriented boxes on image.
    """
    img_copy = img.copy()
    for obj in objects:
        try:
            x1, y1 = obj.p1
            x2, y2 = obj.p2
            cv2.rectangle(img_copy, (x1,y1), (x2,y2), color, thickness)
        except:
            pts = np.array(obj.points, np.int32).reshape((-1,1,2))
            cv2.polylines(img_copy, [pts], True, color, thickness)
    return img_copy

st.title("🛒 Shelf Monitoring Dashboard")

uploaded_file = st.file_uploader("Upload Shelf Image", type=['jpg','png','jpeg'])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    shelves = detect_shelves(img_rgb)
    products = detect_products(img_rgb)

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    st.write(f"### 📏 Detected shelves: {len(shelves)}")
    st.write(f"### 📦 Detected products: {len(products)}")

    for i, shelf in enumerate(shelves):
        shelf_products = []
        for product in products:
            if (product.p1[0] >= shelf.p1[0] and product.p2[0] <= shelf.p2[0] and
                product.p1[1] >= shelf.p1[1] and product.p2[1] <= shelf.p2[1]):
                shelf_products.append(product)

        num_products = len(shelf_products)

        if num_products >= 1:
            avg_width = np.mean([p.p2[0] - p.p1[0] for p in shelf_products])
            shelf_width = shelf.p2[0] - shelf.p1[0]
            total_possible = int(shelf_width / avg_width)
            additional = max(0, total_possible - num_products)
        else:
            additional = 'N/A (no products detected)'

        st.markdown(f"""
        ### 🗂️ Shelf {i+1}
        - **Products detected:** {num_products}
        - **Estimated additional products that can fit:** {additional}
        """)

    img_shelves = draw_boxes(img_rgb, shelves, color=(255,0,0))  # shelves in red
    img_products = draw_boxes(img_shelves, products, color=(0,255,0))  # products in green

    st.image(img_products, caption="Detected Shelves (red) and Products (green)", use_column_width=True)
