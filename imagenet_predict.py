import streamlit as st
from resnet50 import ResNet50
import os


# Display a camera input widget
picture = st.camera_input("Take a picture")

# If a picture is taken, display it and save it
if picture:
    # st.image(picture)
    os.makedirs("test", exist_ok=True)
    with open("test/captured_image.png", "wb") as f:
        f.write(picture.getbuffer())
    # st.success("Picture saved as captured_image.png")
    model = ResNet50(weights="imagenet")
    model.predict(["test/captured_image.png"])
    # st.write(model.labels[0])
    st.metric("Predicted class", model.labels[0], "")

# import streamlit as st
# import sys

# st.write(f"Python version: {sys.version}")
