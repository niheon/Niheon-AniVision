# Import necessary libraries
import streamlit as st  # for creating the UI and handling user input
import cv2  # for reading and processing image files
import numpy as np  # for working with numerical data
import tensorflow as tf  # for loading and using the pre-trained model
from tensorflow.keras.preprocessing import image  # for preprocessing images
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input  # for using MobileNetV2 model for preprocessing
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions  # for decoding the model predictions
from tensorflow.keras import backend as K  # for accessing the model layers and gradients

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

# Run app
if uploaded_file is not None:

    # Read and preprocess image
    # Convert uploaded file to bytes
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode bytes into an image array
    img = cv2.imdecode(img_bytes, 1)
    # Convert the image array to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the image to 224x224 pixels (the input size required by the pre-trained model)
    img_resize = cv2.resize(img, (224,224))
    # Preprocess the resized image using MobileNetV2 preprocessing function
    img_resize = mobilenet_v2_preprocess_input(img_resize)
    # Reshape the preprocessed image to match the input shape of the model
    img_reshape = img_resize[np.newaxis,...]

    # Display image
    st.image(img, channels="RGB")

    # Run prediction
    # Create a button for running the prediction
    predict_img = st.button("Predict Animal")    
    if predict_img:
        # Get the prediction probabilities for each animal class
        prediction = model.predict(img_reshape).argmax()
        try:
            # Look up the predicted animal class from the animal dictionary
            st.title("Predicted Animal is {}".format(animal_dict[prediction]))
        except:
            # Display an error message if the prediction fails
            st.title("Sorry, we are unable to predict")

        # Collect user feedback
        feedback = st.selectbox("Is the predicted animal correct?", ["","Yes", "No"])
        if feedback == "Yes":
            st.write("Thank you for confirming the prediction!", icon="✅")
        elif feedback == "No":
            st.write("We apologize for the incorrect prediction. Please try again with a different image.")
        else:
            st.warning("Please provide feedback on the predicted animal.")

        # Generate Grad-CAM visualization
        # Get the last convolutional layer of the model
        last_conv_layer = model.get_layer('Conv_1')
        # Get the gradient tape context
        tape = K.GradientTape()
        # Start the gradient tape context
        with tape:
            # Feed the image into the model and compute the activations of the last convolutional layer and the prediction
            conv_output, predictions = model(img_reshape)
            # Get the predicted class activation
            pred_class_activation = predictions[:,predicted_class]
        # Compute the gradients of the predicted class activation with respect to the output feature map of the last convolutional layer
        grads = tape.gradient(pred_class_activation, conv_output)
        # Compute the global average pooling of the gradients
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        # Multiply the pooled gradients with the output feature map of the last convolutional layer
        cam = np.multiply(pooled_grads, conv_output)
        # Compute the average of the resulting feature maps along the channel dimension
        cam = np.mean(cam, axis=-1)
        # Apply ReLU activation to the resulting map
        cam = np.maximum(cam, 0)
        # Normalize the map
        cam = cam / np.max(cam)
        # Resize the map to match the size of the input image
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        # Convert the map to 3-channel color heatmap
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        # Blend the heatmap with the input image using an alpha value of 0.5
        blended_image = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
        # Display the blended image with the Grad-CAM heatmap overlay
        st.image(blended_image, channels="RGB")
