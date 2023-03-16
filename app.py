import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras import backend as K

# Set page title
st.set_page_config(
    page_title="Niheon AniVision",
)

# Load pre-trained model
with st.spinner('Loading Model...'):
    model = tf.keras.models.load_model("model/animal.hdf5")

# Set up UI components
st.title("Niheon AniVision")
st.markdown("Niheon AniVision is an AI-powered easy-to-use app for identifying animals (dog, horse, elephant, butterfly, chicken, cat, cow) in a given image. Explore the wonders of wildlife and discover new creatures with just a few clicks. The app may not be able to identify the animal depending on the uploaded image, but it's still a fun way to learn about some animals.")
st.write('Try it with some of the [sample images](https://github.com/niheon/Niheon-AniVision)', unsafe_allow_html=True)

# Upload image file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg","png"])

# Map animal classes
animal_dict = {0: 'dog',
               1: 'horse',
               2: 'elephant',
               3: 'butterfly',
               4: 'chicken',
               5: 'cat',
               6: 'cow'}

# Create session state to store prediction made
def get_or_create_session_state():
    if "session_state" not in st.session_state:
        st.session_state["session_state"] = {}
    return st.session_state["session_state"]

# Run app if image file is uploaded
if uploaded_file is not None:

    # Read and preprocess image
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, (224,224))
    img_resize = mobilenet_v2_preprocess_input(img_resize)
    img_reshape = img_resize[np.newaxis,...]

    # Display image
    st.image(img, channels="RGB")

    # Run prediction
    predict_img = st.button("Predict Animal")
    if predict_img:
        prediction = model.predict(img_reshape).argmax()
        try:
            st.title("Predicted Animal is {}".format(animal_dict[prediction]))
            session_state = get_or_create_session_state()
            session_state["prediction_made"] = True
        except:
            st.title("Sorry, we are unable to predict")

    # Check if the prediction has been made before displaying the feedback selectbox
    session_state = get_or_create_session_state()
    if "prediction_made" in session_state and session_state["prediction_made"]:
        feedback = st.selectbox("Is the predicted animal correct?", ["", "Yes", "No"])
        if feedback == "Yes":
            st.markdown("Thank you for confirming the prediction!")
            session_state["prediction_made"] = False
        elif feedback == "No":
            st.markdown("We apologize for the incorrect prediction. Please try again with a different image.")
            session_state["prediction_made"] = False
        else:
            st.warning("Please provide feedback on the predicted animal.")