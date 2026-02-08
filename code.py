import cv2
import matplotlib.pyplot as plt

# Define image path 
image_path = 'C:\\Users\\54ksh\\OneDrive\\Desktop\\friends.jpg'

try:
    # Read the image in color format (BGR by default in OpenCV)
    img = cv2.imread(image_path)

    # Check if image was read successfully
    if img is None:
        raise FileNotFoundError(f"Image '{image_path}' not found or could not be read.")

    # Convert to grayscale for face detection (more efficient)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces using optimized parameters
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,  # How much the image scale is increased in each iteration
        minNeighbors=5,  # Minimum number of neighbors to reject false positives
        minSize=(150, 150)   # Minimum face size to detect (adjust as needed)
        
    )

    # Draw green rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert image to RGB for Matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a subplot with a specific size for better visualization
    plt.figure(figsize=(10, 6))  # Adjust figure size as desired

    # Display the image with detected faces
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide plot axes for a cleaner presentation
    
    # Add a title to the plot with the number of faces detected
    plt.title(f'Number of faces detected: {len(faces)-1}')
    # Show the image
    plt.show()

except FileNotFoundError as e:
    print(f"Error: {e}")  # Provide an informative error message
