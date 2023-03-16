import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

# Set page title
st.set_page_config(
    page_title="Niheon AniVision",
)

# Load pre-trained model
with st.spinner('Loading Model...'):
    model = tf.keras.models.load_model("model/animal.hdf5")
    
    # Find the last convolutional layer using indexing
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        st.error("Could not find the last convolutional layer in the model.")
    else:
        classifier_layer = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(), model.layers[-1]])

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

def get_gradcam_heatmap(img_array, model, last_conv_layer, classifier_layer):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer(img_tensor)
        tape.watch(last_conv_layer_output)
        preds = classifier_layer(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.matmul(last_conv_layer_output, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

if uploaded_file is not None:

    # Read and preprocess image
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, (224, 224))
    img_resize = mobilenet_v2_preprocess_input(img_resize)
    img_reshape = img_resize[np.newaxis, ...]

    # Display image
    st.image(img, channels="RGB")

    # Run prediction
    predict_img = st.button("Predict Animal")
    if predict_img:
        prediction = model.predict(img_reshape).argmax()
        try:
            st.title("Predicted Animal is {}".format(animal_dict[prediction]))

            # Grad-CAM visualization
            heatmap = get_gradcam_heatmap(img_reshape, model, last_conv_layer, classifier_layer)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img
            superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
            st.image(superimposed_img, caption="Grad-CAM Visualization", use_column_width=True)

        except:
            st.title("Sorry, we are unable to predict")

        feedback = st.selectbox("Is the predicted animal correct?", ["", "Yes", "No"])
        if feedback == "Yes":
            st.markdown("Thank you for confirming the prediction!")
        elif feedback == "No":
            st.markdown("We apologize for the incorrect prediction. Please try again with a different image.")
        else:
            st.warning("Please provide feedback on the predicted animal.")