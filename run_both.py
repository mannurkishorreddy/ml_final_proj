import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# Replace "your-bucket-name" with the name of your GCP Storage bucket
bucket_name = "ml-model-host"

# Replace "your-model.h5" with the name of your .h5 model file
model_file = "20230517_Mid.h5"

# Construct the full model path
model_path = f"gs://{bucket_name}/{model_file}"

# Load the model using the appropriate library function
model = tf.keras.models.load_model(model_path)

# Load the trained model
# model = load_model('20230517_Mid.h5')

# Define a function to preprocess the input image
def preprocess_image(image):
    # Check image shape
    if image.shape[:2] != (224, 224):
        image = cv2.resize(image, (224, 224))
    
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Create a function to predict BMI from an uploaded image
def predict_bmi_from_image(image):
    processed_image = preprocess_image(image)
    bmi_prediction = model.predict(processed_image)
    return bmi_prediction[0][0]

# Create a function to capture live video from the camera and predict BMI
def predict_bmi_live():
    cap = cv2.VideoCapture(0)  # Open the camera
    
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        
        # Preprocess the frame
        processed_image = preprocess_image(frame)
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Draw a bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract the face region from the frame
            face_image = frame[y:y + h, x:x + w]
            
            # Preprocess the face image
            processed_face = preprocess_image(face_image)
            
            # Predict BMI
            bmi_prediction = model.predict(processed_face)
            
            # Display the predicted BMI on the frame
            bmi_text = "Predicted BMI: {:.2f}".format(bmi_prediction[0][0]).zfill(5)
            cv2.putText(frame, bmi_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('BMI Prediction', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows

# Create a Streamlit app
def main():
    st.title("BMI Prediction")

    # Display the options for input
    input_option = st.selectbox("Choose an option:", ("Webcam Input", "Upload Image"))

    if input_option == "Webcam Input":
        st.write("Please wait while the webcam stream is loading...")
        predict_bmi_live()
    elif input_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = np.array(cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1))
            bmi_prediction = predict_bmi_from_image(image)
            st.markdown("<h3 style='text-align: center;'>Predicted BMI: {:.2f}</h3>".format(bmi_prediction), unsafe_allow_html=True)
            st.image(image, channels="BGR")
        else:
            st.write("No image uploaded.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
