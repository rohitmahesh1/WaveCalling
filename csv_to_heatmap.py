import numpy as np
import matplotlib.pyplot as plt
import fire
import os
from PIL import Image
import cv2

def keep_extreme_values(arr, lower, upper):
    # Zero out values within the range [lower, upper]
    arr = arr.copy()
    mask = (arr >= lower) & (arr <= upper)
    arr[mask] = 0
    return arr

def filter_csv_and_generate_heatmap(input_path, output_image_path=None):
    # Load CSV data
    data = np.genfromtxt(input_path, delimiter=',')

    # Filter and binarize the data
    data_extreme_yellow = keep_extreme_values(data, -1e20, 1e16)
    data_extreme_yellow = (abs(data_extreme_yellow) > 0).astype(int)

    # Set default output path
    if output_image_path is None:
        output_image_path = os.path.splitext(input_path)[0] + '.png'

    # Create and save heatmap
    plt.figure(figsize=(8, 6))
    data_extreme_yellow = np.flipud(data_extreme_yellow)
    plt.imshow(abs(data_extreme_yellow), cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(abs(data_extreme_yellow)))
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Heatmap has been saved to {output_image_path}")
    return output_image_path

def filter_csv_and_generate_heatmap_2(input_path, output_image_path=None):
    # Load and filter CSV data
    data = np.genfromtxt(input_path, delimiter=',')
    data_extreme_yellow = keep_extreme_values(data, -1e20, 1e16)
    data_extreme_yellow = (abs(data_extreme_yellow) > 0).astype(int)

    # Set default output path
    if output_image_path is None:
        output_image_path = os.path.splitext(input_path)[0] + '_2.png'

    # Get dimensions
    num_rows, num_cols = data_extreme_yellow.shape

    # Create and save heatmap (origin flipped manually)
    plt.figure(figsize=(8, 6))
    plt.imshow(abs(data_extreme_yellow), cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(abs(data_extreme_yellow)), origin='lower')
    plt.xticks(np.arange(0, num_cols, step=max(1, num_cols // 10)))
    plt.yticks(np.arange(0, num_rows, step=max(1, num_rows // 10)))
    plt.axis('on')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Heatmap has been saved to {output_image_path}")
    return output_image_path, num_rows, num_cols

def filter_csv_and_generate_heatmap_mod(input_path, output_image_path=None):
    # Load and preprocess CSV data
    data = np.genfromtxt(input_path, delimiter=',')
    data_extreme_yellow = keep_extreme_values(data, -1e20, 1e16)
    data_extreme_yellow = (abs(data_extreme_yellow) > 0).astype(int)

    if output_image_path is None:
        output_image_path = os.path.splitext(input_path)[0] + '_d.png'

    # Flip and plot original data
    plt.figure(figsize=(8, 6))
    data = np.flipud(data)
    plt.imshow(data)
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Heatmap has been saved to {output_image_path}")
    return output_image_path

def filter_csv_and_generate_heatmap_grey(input_path, output_image_path=None):
    # Load and filter data
    data = np.genfromtxt(input_path, delimiter=',')
    data_extreme_yellow = keep_extreme_values(data, -1e20, 1e16)
    data_extreme_yellow = (abs(data_extreme_yellow) > 0).astype(int)

    if output_image_path is None:
        output_image_path = os.path.splitext(input_path)[0]

    # Save flipped original data heatmap
    flipped_data = np.flipud(data)
    plt.figure(figsize=(8, 6))
    plt.imshow(flipped_data)
    flipped_data_path = output_image_path + '_data_o.png'
    plt.savefig(flipped_data_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save flipped binary heatmap
    flipped_extreme_yellow = np.flipud(data_extreme_yellow)
    plt.figure(figsize=(8, 6))
    plt.imshow(abs(flipped_extreme_yellow), cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(abs(flipped_extreme_yellow)))
    plt.axis('off')
    plt.savefig(output_image_path + '_extreme_yellow_o.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Flipped heatmap of data_extreme_yellow has been saved to {output_image_path}_extreme_yellow_o.png")

    # Load image and convert green to white, others to black
    img = Image.open(flipped_data_path).convert('RGB')
    img_array = np.array(img)

    lower_green = np.array([0, 100, 0])
    upper_green = np.array([100, 255, 100])

    green_mask = ((img_array[:, :, 0] >= lower_green[0]) & (img_array[:, :, 0] <= upper_green[0]) &
                  (img_array[:, :, 1] >= lower_green[1]) & (img_array[:, :, 1] <= upper_green[1]) &
                  (img_array[:, :, 2] >= lower_green[2]) & (img_array[:, :, 2] <= upper_green[2]))

    img_array[green_mask] = [255, 255, 255]
    img_array[~green_mask] = [0, 0, 0]

    img_grey = Image.fromarray(img_array)
    grey_image_path = output_image_path + '_data_grey_o.png'
    img_grey.save(grey_image_path)
    return output_image_path + '_extreme_yellow_o.png'

def convert_to_grayscale_custom(input_path, output_image_path=None):
    # Load color image
    print("image file:", input_path)
    data = cv2.imread(input_path)

    # Apply filtering and binarization
    data_extreme_yellow = keep_extreme_values(data, -1e20, 1e16)
    data_extreme_yellow = (abs(data_extreme_yellow) > 0).astype(int)

    if output_image_path is None:
        output_image_path = os.path.splitext(input_path)[0] + '_cgs.png'

    # Generate and save heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(abs(data_extreme_yellow), cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(abs(data_extreme_yellow)))
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Heatmap has been saved to {output_image_path}")
    return output_image_path

def convert_to_grayscale(input_path, output_image_path=None):
    # Load and convert image to RGB
    im = cv2.imread(input_path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Define color thresholds
    lower_green = np.array([0, 100, 0])
    upper_green = np.array([150, 255, 150])
    green_mask = cv2.inRange(im_rgb, lower_green, upper_green)

    lower_pink = np.array([200, 100, 100])
    upper_pink = np.array([255, 180, 200])
    lower_purple = np.array([100, 0, 100])
    upper_purple = np.array([200, 100, 255])

    pink_mask = cv2.inRange(im_rgb, lower_pink, upper_pink)
    purple_mask = cv2.inRange(im_rgb, lower_purple, upper_purple)

    # Replace target colors with white
    combined_mask = purple_mask
    im_rgb[combined_mask > 0] = (255, 255, 255)

    # Convert others to grayscale
    non_target_non_green_mask = cv2.bitwise_and(cv2.bitwise_not(combined_mask), cv2.bitwise_not(green_mask))
    im_gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    im_rgb[non_target_non_green_mask > 0] = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2RGB)[non_target_non_green_mask > 0]

    if output_image_path is None:
        output_image_path = os.path.splitext(input_path)[0] + '_cgs.png'

    # Save final result as heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(abs(im_rgb), cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(abs(im_rgb)))
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Heatmap has been saved to {output_image_path}")
    return output_image_path

def convert_to_grayscale_old(image_path, output_image_path=None):
    # Open and convert image to grayscale
    image = Image.open(image_path)
    grayscale_image = image.convert('L')

    if output_image_path is None:
        output_image_path = os.path.splitext(image_path)[0] + '_grayscale.png'

    # Save result
    grayscale_image.save(output_image_path)
    print(f"Grayscale image has been saved to {output_image_path}")
    return output_image_path

if __name__ == '__main__':
    fire.Fire(convert_to_grayscale)
