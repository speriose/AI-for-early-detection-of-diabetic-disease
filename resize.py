from PIL import Image
import os

def resize_images(input_dir, size):
    """
    Resize all images in a directory.

    Args:
        input_dir (str): The path to the directory containing input images.
        size (tuple): A tuple representing the new size (width, height).
    """
    # Iterate over each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
            input_path = os.path.join(input_dir, filename)
            # Resize the image
            original_image = Image.open(input_path)
            resized_image = original_image.resize(size)
            # Generate output path for resized image
            output_path = os.path.join(input_dir, f"resized_{filename}")
            # Save the resized image
            resized_image.save(output_path)
            # Delete the original image
            os.remove(input_path)

# Example usage:
input_directory = r"C:\Users\Mohamed\Desktop\dataset_3\Diabetic retinopathy\Moderate"
new_size = (64, 64)  # Width, Height

resize_images(input_directory, new_size)
