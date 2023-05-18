import cv2
import streamlit as st

def main():
    st.title("Camera Feed")
    
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        st.error("Unable to open the camera.")
        return
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        if ret:
            # Display the frame in Streamlit
            st.image(frame, channels="BGR")
        
        # Check if the 'q' key is pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
