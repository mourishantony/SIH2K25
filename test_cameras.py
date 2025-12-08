import cv2
cap = cv2.VideoCapture(0)
print("Camera 0 opened:", cap.isOpened())
cap.release()

cap1 = cv2.VideoCapture(1)
print("Camera 1 opened:", cap1.isOpened())
cap1.release()
