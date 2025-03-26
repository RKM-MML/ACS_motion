import cv2
import numpy as np
import glob
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T
import os

# Load Mask R-CNN model once globally
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
model.to(device)

def load_images_from_folder(folder_path):
    """Load all JPG images from the specified folder."""
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

def preprocess_image(image):
    """Convert image to RGB, transform to tensor, and send to device."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    return transform(image_rgb).unsqueeze(0).to(device)

def get_object_mask(image):
    """Use Mask R-CNN to get an object mask for the image."""
    with torch.no_grad():
        image_tensor = preprocess_image(image)
        prediction = model(image_tensor)[0]
        if 'masks' not in prediction or len(prediction['masks']) == 0:
            raise ValueError("No valid masks detected")
        threshold = 0.5
        if prediction['scores'][0] < threshold:
            raise ValueError("No masks with a score above threshold")
        mask = prediction['masks'][0, 0].mul(255).byte().cpu().numpy()
        return mask

def traditional_masking(image):
    """Fallback masking using Canny edges and morphology."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.inRange(dilated_edges, 1, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def mask_image_with_rcnn(image):
    """Apply Mask R-CNN masking; fall back to traditional masking if necessary."""
    try:
        mask = get_object_mask(image)
    except ValueError:
        mask = traditional_masking(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask

def focus_stack(images):
    """Perform focus stacking on a list of images.
    
    For each pixel, choose the pixel from the image with the highest Laplacian response.
    Returns:
        stacked_image: the final composite image.
        focus_indices: array indicating which image index contributed each pixel.
    """
    stack_shape = images[0].shape[:2]
    focus_measure = np.zeros(stack_shape)
    focus_indices = np.zeros(stack_shape, dtype=int)

    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mask = laplacian > focus_measure
        focus_measure[mask] = laplacian[mask]
        focus_indices[mask] = i

    stacked_image = np.zeros_like(images[0])
    for y in range(stack_shape[0]):
        for x in range(stack_shape[1]):
            stacked_image[y, x] = images[focus_indices[y, x]][y, x]

    return stacked_image, focus_indices

def create_depth_map(focus_indices, layer_distance):
    """Create a depth map by multiplying focus indices with the distance between layers."""
    return focus_indices * layer_distance

def depth_map_to_point_cloud(depth_map, image, xy_scale=1.0, z_scale=1.0):
    """Convert a depth map and corresponding image to a point cloud.
    
    Returns:
        points: Nx3 numpy array of (x, y, z) coordinates.
        colors: Nx3 numpy array of normalized colors.
    """
    h, w = depth_map.shape
    points = []
    colors = []
    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * z_scale
            if z != 0:
                points.append([x * xy_scale, y * xy_scale, z])
                color = image[y, x] / 255.0
                colors.append(color)
    return np.array(points), np.array(colors)

def calculate_dimensions(points):
    """Calculate the dimensions (length, breadth, height) of the point cloud."""
    if len(points) == 0:
        return 0, 0, 0
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    return x_max - x_min, y_max - y_min, z_max - z_min

def process_focus_stack(images, progress_callback=None, layer_distance=100, xy_scale=0.01, z_scale=0.001):
    """
    Process a list of images to perform focus stacking, masking, depth mapping,
    and point cloud reconstruction.

    Args:
        images (list): List of images loaded by OpenCV.
        progress_callback (function): Optional function to update progress (called with current index).
        layer_distance (float): Distance between image layers for depth mapping.
        xy_scale (float): Scale factor for x and y dimensions.
        z_scale (float): Scale factor for z dimension.
    
    Returns:
        stacked_image (numpy.ndarray): The composite focus stacked image.
        point_cloud (tuple): A tuple (points, colors) representing the point cloud.
        dimensions (tuple): (length, breadth, height) of the reconstructed object.
    """
    images_cleaned = []
    for i, img in enumerate(images):
        masked_img, _ = mask_image_with_rcnn(img)
        images_cleaned.append(masked_img)
        if progress_callback:
            progress_callback(i + 1)
    if not images_cleaned:
        raise ValueError("No valid images after processing.")
    
    stacked_image, focus_indices = focus_stack(images_cleaned)
    depth_map = create_depth_map(focus_indices, layer_distance)
    point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale, z_scale)
    dimensions = calculate_dimensions(point_cloud)
    
    return stacked_image, (point_cloud, colors), dimensions

# For testing purposes only
if __name__ == '__main__':
    folder_path = input("Enter folder path: ")
    imgs = load_images_from_folder(folder_path)
    stacked, point_cloud_data, dims = process_focus_stack(imgs)
    print("Dimensions (L, B, H):", dims)
