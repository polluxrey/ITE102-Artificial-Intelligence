import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pickle as pk
import Orange

WIDTH = HEIGHT = 28
THRESHOLD = 100

# Change the page title and icon according to your information
# Set the page_title using this format: [Last_Name]_[Section]_Lab_Exercise_2
st.set_page_config(page_title="Rey_4A_Lab_Exercise_2", page_icon=":two:")

st.title(":hash: :blue[Digit]:blue-background[Detect] :mag:", anchor=False)

st.divider()

st.header("Upload Your Handwritten Number", anchor=False)

multi = """Upload a **SQUARE IMAGE** of a handwritten number on a **WHITE BACKGROUND** for best results!

**P.S.** Please do not break the app! :sob:
"""
st.markdown(multi)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg"])

if uploaded_file:
    st.divider()
    col1, col2, col3 = st.columns(3)
    col2.image(uploaded_file)

    data = []
    img = Image.open(uploaded_file)
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((WIDTH, HEIGHT), resample=Image.LANCZOS)
    img_arr = np.array(img)
    img_arr = np.where(img_arr > THRESHOLD, img_arr, 0)
    img_arr = img_arr.flatten()
    data.append(img_arr)
    data = np.array(data)

    print(data)

    # Change "Rey_4A_Lab_Exercise_2_Model.pkcls" to file name of your model
    with open("Rey_4A_Lab_Exercise_2_Model.pkcls", "rb") as file:
        model = pk.load(file)
        predictions = model.predict(data)
        st.markdown(
            f"<h1 style='text-align: center;'>The model predicts this is {int(predictions[0][0])}!</h1>",
            unsafe_allow_html=True,
        )
