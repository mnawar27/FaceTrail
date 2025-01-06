import cv2

cap = cv2.VideoCapture(0)  # Use 0 or the appropriate camera index
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
