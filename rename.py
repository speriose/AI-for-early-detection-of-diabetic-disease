from PIL import Image
import os

def convert_and_replace_with_jpg(folder_path):
    """
    Converts all image files in a folder to .jpg format and replaces the originals.

    Parameters:
        folder_path (str): Path to the folder containing the image files.

    Returns:
        None
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is valid
        if os.path.isfile(file_path):
            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Convert to RGB (required for JPEG format)
                    img = img.convert("RGB")

                    # Create a new file path with .jpg extension
                    new_file_path = os.path.splitext(file_path)[0] + ".jpg"

                    # Save as .jpg
                    img.save(new_file_path, "JPEG")
                    print(f"Converted: {filename} -> {os.path.basename(new_file_path)}")

                # Remove the original file if it wasn't already a .jpg
                if new_file_path != file_path:
                    os.remove(file_path)
                    print(f"Removed original file: {filename}")

            except Exception as e:
                print(f"Skipping {filename}: {e}")

if __name__ == "__main__":
    folder_path = r'C:\Users\Mohamed\Desktop\Anaconda workspace\train\diabetic_retinopathy'
    convert_and_replace_with_jpg(folder_path)
