import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from keras.models import load_model
from streamlit_option_menu import option_menu
from keras.preprocessing.image import img_to_array


st.set_page_config(page_title="Dashboard",layout="wide",)
st.sidebar.image(r'assets/logo.png', caption='WELLCOME')
mappings = { 0: 'Finger millet', 1: 'Pearl millet' }


def Upload():
    st.title("Upload a single image")
    st.markdown("---")
    st.header("Image File Upload")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    col1 , col2, = st.columns(2)
    predict = False
    model_image = np.array([])   
    with col1:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            model_image = image.copy()
            image = cv2.resize(np.array(image),(400,400)) 
            col1=st.image(image, caption='Uploaded Image.')
            predict = st.button(label="Click Me for Prediction")

    with col2:
        if predict:
            model_image = cv2.resize(np.array(model_image),(600,600))
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'model_checkpoint_6.h5')
            model = load_model(model_path,compile=False)
            img_array = img_to_array(model_image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)
            data = {
                'Predicted Index': [predicted_class],
                'Mapped Class': [mappings[predicted_class[0]]]
            }
            col2 = st.dataframe(data,550,50)

def Uploads():
    st.title("Upload images...")
    st.markdown("---")
    uploaded_files = st.file_uploader("Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    predict = False
    model_image = np.array([]) 
    if uploaded_files:
        st.write("Uploaded Images:")
        num_columns = 3 
        images = [Image.open(file) for file in uploaded_files]

        num_rows = len(images) // num_columns + int(len(images) % num_columns != 0)

        for row in range(num_rows):
            cols = st.columns(num_columns)
            for col, image in zip(cols, images[row*num_columns:(row+1)*num_columns]):
                with col:
                    image = cv2.resize(np.array(image),(400,400)) 
                    st.image(image) 
        count = 0
        predict = st.button(label="Click Me for Prediction")
        
        index = []
        category = []
        
        if predict:
            progress_bar = st.progress(0)
            print(uploaded_files)
            for idx, file in enumerate(uploaded_files):
                image = Image.open(file)
                model_image = cv2.resize(np.array(image),(600,600))
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, 'model_checkpoint_6.h5')
                model = load_model(model_path,compile=False)
                img_array = img_to_array(model_image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)
                index.append(predicted_class)
                category.append(mappings[predicted_class[0]])
                progress = (idx + 1) / len(images)
                progress_bar.progress(progress)
            # st.text(f'Processed {idx + 1} of {len(images)} files')
               
            data = {
                'Predicted Index': index,
                'Mapped Class': category,
            }
            st.dataframe(data,600) 
      
def Home():
    st.header('Hello Welcome')
    pass

def sideBar():
    with st.sidebar:
        selected = option_menu(
           menu_title = 'Main Menu',
           options= ['Home','Upload','Uploads'],
           icons=["house","file","folder"], 
           default_index = 0,
           menu_icon = 'cast'  
        ) 
    if selected == 'Home':
        Home()
    # if selected == 'Camera':
    #     Camera()
    if selected == 'Upload':
        Upload()
    if selected == 'Uploads':
        Uploads()

def main():
    sideBar()

if __name__ == '__main__':
    main()
