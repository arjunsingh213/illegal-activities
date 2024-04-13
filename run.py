import cv2
import numpy as np
import streamlit as st
from PIL import Image

@st.cache(allow_output_mutation=True)
def get_predictor_model():
    from model import Model
    model = Model()
    return model

header = st.container()
model = get_predictor_model()

with header:
    st.title('Hello!')
    st.text('Using this app you can classify whether there is fight on a street? or fire? or car crash? or everything is okay?')

uploaded_file = st.file_uploader("Choose an image or video file...")

if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type.startswith("image/"):
        # Handle image file
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        label_text = model.predict(image=image_np)['label'].title()
        st.write(f'Predicted label for the image is: **{label_text}**')
        st.image(image_np)

    elif file_type.startswith("video/"):
        # Handle video file
        st.write("Uploaded file is a video.")
        
        # Use OpenCV to read the video
        video_file_path = uploaded_file.name  # Use the file name directly
        video_capture = cv2.VideoCapture(video_file_path)

        if not video_capture.isOpened():
            st.error("Could not open video file.")
        else:
            # Process video frames
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                
                if not ret:
                    break  # Exit the loop when no more frames

                # Convert the frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Use the model to predict the frame
                label_text = model.predict(image=frame_rgb)['label'].title()
                
                st.write(f'Predicted label for the current frame: **{label_text}**')
                st.image(frame_rgb)

            video_capture.release()
