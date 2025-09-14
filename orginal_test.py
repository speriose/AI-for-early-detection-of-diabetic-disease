import os
import smtplib
import serial
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

##SerialObj = serial.Serial('COM3', baudrate=115200, bytesize=8, parity='N', stopbits=1)    

#serial_port = 'COM3'
#baud_rate = 115200  # Match the baud rate set in your ESP32

# Load the saved model
model_1 = load_model(r'D_R.h5')

# Define the target shape for input images
target_shape = (64, 64)  # Adjust this to match the target shape used during training

# Define your class labels
classes_1 = ['Moderate', 'No_DR']
#classes_2 = ['acute_lymphocytic', 'chronic_lymphocytic']
#classes_3 = ['acute_myeloid', 'chronic_myeloid']

# Function to preprocess and classify an image file
def test_image(file_path, model):
    # Load and preprocess the image
    img = load_img(file_path, target_size=target_shape)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]

    # Make predictions
    predictions = model.predict(img_array)

    # Get the class probabilities
    class_probabilities = predictions[0]

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    return class_probabilities, predicted_class_index

###############################################################################

######### Define the smtp gmail API#########
email = "mah87146@gmail.com"               
app_password = "exis ccsz mfqu fkbt"       
message = "Hello, Your test result is : "   
############################################

def send_mail(message):
    try:
        # Email details (edit it for yours)
        sender_email = "mah87146@gmail.com"
        receiver_email = "mah87146@gmail.com"
        password = "jqiz yrvo nfja fqrq"

        # Create message object
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = "Image Classification Result"

        # Add message body
        body = message
        msg.attach(MIMEText(body, 'plain'))

        # Connect to Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        # Login to your Gmail account
        server.login(sender_email, password)

        # Send email
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print("[*] Email sent successfully!")

        # Close connection
        server.quit()

    except Exception as e:
        print("An error occurred while sending email:", str(e))


# Test the image file

while (True):
    try:
        test_image_file = input("[+] Enter the photo path here >> ")

        class_probabilities, predicted_class_index = test_image(test_image_file, model_1)

        # Display results for all classes
        for i, class_label in enumerate(classes_1):
            probability = class_probabilities[i]
            print(f'Class: {class_label}, Probability: {probability:.4f}')

        # Calculate and display the predicted class
        predicted_class_1 = classes_1[predicted_class_index]
        print(f'The image is classified as: {predicted_class_1}')

        
            
        # Open the serial connection
        #ser = serial.Serial(serial_port, baud_rate, timeout=1)

           

            ### Send message to ESP32
            #ser.write(predicted_class_3.encode('utf-8'))
            #print(f"Sent: {predicted_class_3}")

            # Close the serial connection
            #ser.close()
            
            
            #send_mail(message + predicted_class_3)### SEND MAIL
            #print("[*] Email sent succesfully ! ")
    except:
        print("\n[*] Keyboard intrupt (Ctrl + C), exiting . . . ")
        exit()