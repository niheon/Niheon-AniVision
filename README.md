# Niheon AniVision

![](https://github.com/niheon/Niheon-AniVision/blob/main/Niheon_AniVision_Demo_Github_2.gif =250x250)

## Description
Niheon AniVision is an AI-powered app that can identify the an animal in any uploaded image. With just a few clicks, you can explore the wonders of wildlife and discover new creatures.

The app uses a pre-trained machine learning model, MobileNetV2 model (which has been trained on a large dataset of animal images) to identify an animal in an uploaded image.

When an image is uploaded, Niheon AniVision reads and preprocesses the image using OpenCV and TensorFlow. The image is then resized to 224x224 pixels and normalized to the range [-1,1]. The preprocessed image is then passed through the MobileNetV2 model to generate a prediction.

The predicted animal is displayed on the screen, along with the uploaded image.
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Requirements
Before running the project, make sure you have the following installed:

Python 3, Streamlit, OpenCV, Numpy, TensorFlow.

You can use pip to install the requirements text.

## Installing

1. Clone the repository:

```bash
git clone https://github.com/niheon/Niheon-AniVision.git
```

2. Change to the project directory:

```bash
cd Niheon-AniVision
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```
## Usage

1. Run the app using the following command:

```bash
streamlit run app.py
```
This will start the app on your local server, and you can access it by visiting localhost in your web browser.

2. Upload an image file by clicking the "Choose an image file" button.

Click the "Predict Animal" button to get a prediction of the animal in the uploaded image.

## Demo
Try a live demo of the [app](https://niheon-niheon-anivision-app-hzgh85.streamlit.app/).

## Built With
- Streamlit - The web framework used
- OpenCV - Library for computer vision tasks
- NumPy - Library for numerical computing
- TensorFlow - Open-source machine learning framework

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- MobileNetV2 - The pre-trained model used in this app.
- Animal-10 - The dataset used to train the model.
