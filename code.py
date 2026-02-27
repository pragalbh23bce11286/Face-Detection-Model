import cv2
import matplotlib.pyplot as plt


image_path = 'C:\\Users\\54ksh\\OneDrive\\Desktop\\friends.jpg'

try:
   
    img = cv2.imread(image_path)

    
    if img is None:
        raise FileNotFoundError(f"Image '{image_path}' not found or could not be read.")

    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

   
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,  # How much the image scale is increased in each iteration
        minNeighbors=5,  # Minimum number of neighbors to reject false positives
        minSize=(150, 150)   # Minimum face size to detect (adjust as needed)
        
    )

    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

   
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   
    plt.figure(figsize=(10, 6))  # Adjust figure size as desired

    
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide plot axes for a cleaner presentation
    
   
    plt.title(f'Number of faces detected: {len(faces)-1}')
   
    plt.show()

except FileNotFoundError as e:
    print(f"Error: {e}")  # Provide an informative error message
