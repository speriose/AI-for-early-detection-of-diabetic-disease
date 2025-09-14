import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from glob import glob

# === Change these paths to your dataset folders ===
class1_dir = 'train/dataset_3/Diabetic retinopathy/Moderate'
class2_dir = 'train/dataset_3/Diabetic retinopathy/No_DR'

# Get image file paths
class1_files = glob(os.path.join(class1_dir, '*.png'))
class2_files = glob(os.path.join(class2_dir, '*.png'))

print(f'Class 1: {len(class1_files)} images')
print(f'Class 2: {len(class2_files)} images')

# Create output folder (optional)
output_dir = os.getcwd()  # current working directory

# ====================
# 1️⃣ Class distribution
# ====================
counts = [len(class1_files), len(class2_files)]
labels = ['Class 1', 'Class 2']

plt.figure(figsize=(5,5))
plt.bar(labels, counts, color=['skyblue', 'lightcoral'])
plt.title('Class Distribution')
plt.ylabel('Number of Images')
plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
plt.show()

# ====================
# 2️⃣ Sample images
# ====================
def plot_sample_images(image_files, title, save_name, n=5):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        img = Image.open(image_files[i])
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(title)
    plt.savefig(os.path.join(output_dir, save_name))
    plt.show()

plot_sample_images(class1_files, 'Sample Images from Class 1', 'samples_class1.png')
plot_sample_images(class2_files, 'Sample Images from Class 2', 'samples_class2.png')

# ====================
# 3️⃣ Pixel intensity histogram
# ====================
def plot_intensity_histogram(image_files, title, save_name):
    pixels = []
    for file in image_files[:200]:  # limit for speed
        img = Image.open(file).convert('L')
        pixels.extend(np.array(img).flatten())
    plt.figure(figsize=(6,4))
    plt.hist(pixels, bins=50, color='gray', alpha=0.7)
    plt.title(f'Pixel Intensity Histogram: {title}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, save_name))
    plt.show()

plot_intensity_histogram(class1_files, 'Class 1', 'intensity_hist_class1.png')
plot_intensity_histogram(class2_files, 'Class 2', 'intensity_hist_class2.png')

# ====================
# 4️⃣ Average image per class
# ====================
def compute_mean_image(image_files):
    sum_img = np.zeros((64, 64, 3), dtype=np.float32)
    for file in image_files:
        img = Image.open(file).resize((64, 64)).convert('RGB')
        sum_img += np.array(img, dtype=np.float32)
    mean_img = sum_img / len(image_files)
    return np.uint8(mean_img)

mean_img_class1 = compute_mean_image(class1_files)
mean_img_class2 = compute_mean_image(class2_files)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(mean_img_class1)
plt.title('Mean Image - Class 1')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(mean_img_class2)
plt.title('Mean Image - Class 2')
plt.axis('off')
plt.savefig(os.path.join(output_dir, 'mean_images.png'))
plt.show()

# ====================
# 5️⃣ RGB channel histogram
# ====================
def plot_rgb_histogram(image_files, title, save_name):
    r_vals, g_vals, b_vals = [], [], []
    for file in image_files[:200]:
        img = Image.open(file).convert('RGB')
        arr = np.array(img)
        r_vals.extend(arr[:,:,0].flatten())
        g_vals.extend(arr[:,:,1].flatten())
        b_vals.extend(arr[:,:,2].flatten())
    
    plt.figure(figsize=(10,4))
    plt.hist(r_vals, bins=50, color='red', alpha=0.5, label='Red')
    plt.hist(g_vals, bins=50, color='green', alpha=0.5, label='Green')
    plt.hist(b_vals, bins=50, color='blue', alpha=0.5, label='Blue')
    plt.title(f'RGB Channel Histogram: {title}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, save_name))
    plt.show()

plot_rgb_histogram(class1_files, 'Class 1', 'rgb_hist_class1.png')
plot_rgb_histogram(class2_files, 'Class 2', 'rgb_hist_class2.png')
