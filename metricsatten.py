import os
import glob
import cv2
import numpy as np
import torch
import threading
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame
from tkinter import ttk
import open3d as o3d
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ================================
# PART 1: 3D Point Cloud Reconstruction
# ================================

# Load Mask R-CNN model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
model.to(device)

def load_images_from_folder(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    return transform(image_rgb).unsqueeze(0).to(device)

def get_object_mask(image):
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.inRange(dilated_edges, 1, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def mask_image_with_rcnn(image):
    try:
        mask = get_object_mask(image)
    except ValueError:
        mask = traditional_masking(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask

def focus_stack(images):
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
    return focus_indices * layer_distance

def depth_map_to_point_cloud(depth_map, image, xy_scale=1.0, z_scale=1.0):
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
    if len(points) == 0:
        return 0, 0, 0
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    return x_max - x_min, y_max - y_min, z_max - z_min

def extract_point_cloud(folder_path):
    """
    Given a folder of images for one object, reconstruct the 3D point cloud.
    """
    images = load_images_from_folder(folder_path)
    images_cleaned = []
    for img in images:
        masked_img, _ = mask_image_with_rcnn(img)
        images_cleaned.append(masked_img)
    stacked_image, focus_indices = focus_stack(images_cleaned)
    depth_map = create_depth_map(focus_indices, layer_distance=100)
    # Scale factors (tune as needed)
    point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale=0.01, z_scale=0.001)
    # Downsample the point cloud for consistency
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    points = np.asarray(pcd.points)
    return points

# ================================
# PART 2: Statistical Shape Model (SSM) Construction and Evaluation
# ================================

def load_shapes_from_root(root_folder):
    """
    Assume that each subfolder in root_folder contains images for one shape.
    Returns a list of point clouds.
    """
    shape_folders = [os.path.join(root_folder, d) for d in os.listdir(root_folder)
                     if os.path.isdir(os.path.join(root_folder, d))]
    shapes = []
    for folder in shape_folders:
        try:
            pts = extract_point_cloud(folder)
            # Ensure a fixed number of points per shape (e.g., via random sampling)
            n_points = 1000  # Adjust as needed
            if pts.shape[0] >= n_points:
                idx = np.random.choice(pts.shape[0], n_points, replace=False)
                pts = pts[idx, :]
            else:
                # If not enough points, repeat some points
                idx = np.random.choice(pts.shape[0], n_points, replace=True)
                pts = pts[idx, :]
            shapes.append(pts)
        except Exception as e:
            print(f"Error processing {folder}: {e}")
    return shapes

def align_shapes(shapes):
    """
    A simple alignment: subtract the mean from each shape.
    In practice, you may want to perform Procrustes alignment.
    Each shape is an (n_points, 3) array.
    """
    aligned = []
    for s in shapes:
        aligned.append(s - np.mean(s, axis=0))
    return aligned

def build_ssm(shapes):
    """
    Build a Statistical Shape Model using PCA.
    Each shape is flattened into a vector.
    """
    shapes = align_shapes(shapes)
    n_shapes = len(shapes)
    n_points = shapes[0].shape[0]
    data = np.array([s.flatten() for s in shapes])  # Shape: (n_shapes, n_points*3)
    pca = PCA(n_components=min(n_shapes, n_points * 3))
    pca.fit(data)
    return pca, data

def plot_cumulative_variance(pca):
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.figure()
    plt.plot(cumsum, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Compactness of SSM')
    plt.grid(True)
    plt.show()

def compute_generalization(data, n_components):
    """
    Compute leave-one-out reconstruction error as a measure of generalization.
    For each shape, leave it out, build PCA on remaining shapes, reconstruct it, and compute RMSE/MAE.
    """
    errors_rmse = []
    errors_mae = []
    n_shapes = data.shape[0]
    for i in range(n_shapes):
        train_data = np.delete(data, i, axis=0)
        test_data = data[i]
        pca = PCA(n_components=n_components)
        pca.fit(train_data)
        coeff = pca.transform(test_data.reshape(1, -1))
        recon = pca.inverse_transform(coeff)
        rmse = np.sqrt(mean_squared_error(test_data, recon.flatten()))
        mae = mean_absolute_error(test_data, recon.flatten())
        errors_rmse.append(rmse)
        errors_mae.append(mae)
    return np.mean(errors_rmse), np.mean(errors_mae)

def compute_specificity(pca, data, n_samples=20):
    """
    Generate random shapes from the SSM and compute the error with the closest training shape.
    """
    synthesized = []
    n_coeff = pca.n_components_
    for _ in range(n_samples):
        # sample coefficients from a normal distribution scaled by eigenvalues
        coeff = np.random.randn(n_coeff) * np.sqrt(pca.explained_variance_)
        shape_vec = pca.mean_ + np.dot(coeff, pca.components_)
        synthesized.append(shape_vec)
    synthesized = np.array(synthesized)
    errors = []
    for s in synthesized:
        # compute the minimum RMSE between the synthesized shape and all training shapes
        rmses = [np.sqrt(mean_squared_error(s, d)) for d in data]
        errors.append(min(rmses))
    return np.mean(errors)

def plot_principal_components(pca, data, component_idx=0):
    """
    Visualize the effect of the first principal component by adding/subtracting its standard deviation.
    """
    mean_shape = pca.mean_.reshape(-1, 3)
    std = np.sqrt(pca.explained_variance_[component_idx])
    comp = pca.components_[component_idx]
    comp = comp.reshape(-1, 3)
    shape_plus = mean_shape + std * comp
    shape_minus = mean_shape - std * comp

    fig = plt.figure(figsize=(12, 4))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(mean_shape[:, 0], mean_shape[:, 1], mean_shape[:, 2], c='gray')
    ax1.set_title('Mean Shape')
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(shape_plus[:, 0], shape_plus[:, 1], shape_plus[:, 2], c='green')
    ax2.set_title(f'+1 Std along PC{component_idx+1}')
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(shape_minus[:, 0], shape_minus[:, 1], shape_minus[:, 2], c='red')
    ax3.set_title(f'-1 Std along PC{component_idx+1}')
    
    plt.show()

# ================================
# PART 3: Graphical User Interface (Optional)
# ================================

class PointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title('3D Point Cloud Reconstruction & SSM Evaluation Tool')
        self.master.geometry('800x600')
        self.frame = Frame(master)
        self.frame.pack()
        
        Label(self.frame, text="3D Point Cloud Reconstruction & SSM Evaluation", font=("Arial", 18, "bold")).pack()
        Button(self.frame, text='Select Root Folder (Multiple Shapes)', command=self.upload_root_folder).pack(pady=10)
        Button(self.frame, text='Run SSM & Metrics', command=self.run_ssm_processing).pack(pady=10)
        self.metric_label = Label(self.frame, text="", justify="left")
        self.metric_label.pack(pady=10)
        self.progress_bar = ttk.Progressbar(self.frame, length=300, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.root_folder = None

    def upload_root_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.root_folder = folder_path
            messagebox.showinfo("Info", f"Root folder selected: {folder_path}")
    
    def run_ssm_processing(self):
        if not self.root_folder:
            messagebox.showwarning("Warning", "Please select a root folder first.")
            return
        threading.Thread(target=self.run_ssm_thread, daemon=True).start()
    
    def run_ssm_thread(self):
        shapes = load_shapes_from_root(self.root_folder)
        if len(shapes) == 0:
            self.metric_label.config(text="Error: No valid shapes found.")
            return
        pca, data = build_ssm(shapes)
        # Plot cumulative variance (Compactness)
        plot_cumulative_variance(pca)
        
        # Compute Generalization Error for a chosen number of components (e.g., 10)
        gen_rmse, gen_mae = compute_generalization(data, n_components=10)
        # Compute Specificity Error
        spec_error = compute_specificity(pca, data, n_samples=20)
        
        # Plot the effect of the first principal component
        plot_principal_components(pca, data, component_idx=0)
        
        metrics_text = (
            f"Number of Shapes: {len(shapes)}\n"
            f"Cumulative Variance (first 10 PCs): {np.sum(pca.explained_variance_ratio_[:10]):.2f}\n"
            f"Generalization RMSE: {gen_rmse:.4f}\n"
            f"Generalization MAE: {gen_mae:.4f}\n"
            f"Specificity Error: {spec_error:.4f}\n"
        )
        self.metric_label.config(text=metrics_text)

if __name__ == "__main__":
    # To run the GUI version:
    root = Tk()
    app = PointCloudApp(root)
    root.mainloop()
    
    # Alternatively, to run without the GUI, uncomment the following lines:
    # root_folder = r'path_to_root_folder_with_multiple_shape_folders'
    # shapes = load_shapes_from_root(root_folder)
    # pca, data = build_ssm(shapes)
    # plot_cumulative_variance(pca)
    # gen_rmse, gen_mae = compute_generalization(data, n_components=10)
    # spec_error = compute_specificity(pca, data, n_samples=20)
    # print(f"Generalization RMSE: {gen_rmse}\nGeneralization MAE: {gen_mae}\nSpecificity: {spec_error}")
    # plot_principal_components(pca, data, component_idx=0)
