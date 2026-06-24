# detect.py

from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

def run_inference_and_generate_report(image_path):
    model = YOLO('yolov8n.pt')  # Generic pretrained model

    results = model(image_path)

    # Load image for drawing
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            label = f"Defect ({conf:.2f})"
            draw.rectangle(xyxy, outline="red", width=3)
            draw.text((xyxy[0], xyxy[1] - 10), label, fill="red", font=font)

    # Save output image
    output_dir = os.path.join("static", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, basename)
    image.save(output_path)

    # Generate report
    report_dir = os.path.join("static", "reports")
    os.makedirs(report_dir, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(report_dir, f"report_{now}.txt")
    
    detections_found = False 
    
    with open(report_path, "w") as f:
        f.write(f"Image: {basename}\n")
        f.write(f"Inference Time: {now}\n")
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                xyxy = [round(x.item(), 2) for x in box.xyxy[0]]
                f.write(f"Detected Defect - Confidence: {conf:.2f}, Box: {xyxy}\n")

        if not detections_found:
            f.write("No defects detected.\n")

    return output_path, report_path
