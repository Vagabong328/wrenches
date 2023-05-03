import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

import pandas as pd

# File path of the Excel file
file_path = r'C:\Users\Администратор\Downloads\Wrenches.xlsx'

# Sheet name to read
sheet_name = 'Лист1'

# Read Excel file
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Display the data in the DataFrame
print(df)

# User input prompt to request information based on ID

ZONE_POLYGON = np.array([
    [0, 0],
    [0.6, 0],
    [0.6, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1150, 720],
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            #f"{model.model.names[class_id]} {confidence:0.2f}"
            "wrenches"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # https://medium.com/analytics-vidhya/gaussian-blurring-with-python-and-opencv-ba8429eb879b
        blur = cv2.GaussianBlur(gray, (41,41), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        print(cv2.THRESH_BINARY, cv2.THRESH_OTSU)

        # Morphological transformations https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # Find contours https://learnopencv.com/contour-detection-using-opencv-python-c/
        # CHAIN_APPROX_SIMPLE
        # Алгоритм сжимает горизонтальные, вертикальные и диагональные сегменты вдоль контура и оставляет только их конечные точки.
        # Это означает, что любая из точек вдоль прямых путей будет отклонена, и у нас останутся только конечные точки.
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        i=1
        for c in cnts:
            # Find perimeter of contour
            perimeter = cv2.arcLength(c, True)
            # Perform contour approximation
            approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
            # print(approx)
            # We assume that if the contour has more than a certain
            # number of verticies, we can make the assumption
            # that the contour shape is a circle
            if len(approx) > 0:

                # Obtain bounding rectangle to get measurements
                x, y, w, h = cv2.boundingRect(c)

                # Find measurements
                diameter = w
                radius = w / 2

                # Find centroid
                M = cv2.moments(c)

                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                # print(cX,cY)
                # Draw the contour and center of the shape on the image
                # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),4)
                cv2.drawContours(frame, [c], 0, (36, 255, 12), 1)
                # cv2.circle(image, (cX, cY), 15, (320, 159, 22), -1)

                # Draw line and diameter information
                # cv2.line(image, (x, y + int(h/2)), (x + w, y + int(h/2)), (156, 188, 24), 3)
                # cv2.putText(image, "Diameter: {}".format(diameter), (cX - 150, cY - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (156, 188, 24), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX

        # org
                org = (cX, cY)

        # fontScale
                fontScale = 1

        # Blue color in BGR
                color = (255, 0, 0)

        # Line thickness of 2 px
                thickness = 2

        # Using cv2.putText() method
                frame = cv2.putText(frame,str(i), org, font, fontScale, color, thickness, cv2.LINE_AA)
                i=i+1
        # Displaying the image
                cv2.imshow("yolov8", frame)

                if (cv2.waitKey(30) == 27):
                    break
            request = i

            try:
                id = int(request)
                result = df[df['id'] == id]
                if result.empty:
                    print("id not found.")
                else:
                    print(result)
            except ValueError:
                print("Invalid input. Please enter a valid integer id.")



if __name__ == "__main__":
    main()