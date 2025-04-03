# import sys
# import cv2
# import numpy as np
# import glob
# import torch
# import threading
# import time
# import json
# from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
# from torchvision import transforms as T
# from tkinter import Tk, Label, Button, filedialog, messagebox, Frame
# from tkinter import ttk
# import open3d as o3d
# import os
# import gc

# # Load Mask R-CNN model
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
# model.eval()
# model.to(device)

# def load_images_from_folder(folder_path):
#     image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
#     if not image_files:
#         raise FileNotFoundError(f"No images found in the folder: {folder_path}")
#     images = [cv2.imread(img_file) for img_file in image_files]
#     return images

# def preprocess_image(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     transform = T.Compose([T.ToTensor()])
#     return transform(image_rgb).unsqueeze(0).to(device)

# def get_object_mask(image):
#     with torch.no_grad():
#         image_tensor = preprocess_image(image)
#         prediction = model(image_tensor)[0]

#         if 'masks' not in prediction or len(prediction['masks']) == 0:
#             raise ValueError("No valid masks detected")

#         threshold = 0.5
#         if prediction['scores'][0] < threshold:
#             raise ValueError("No masks with a score above threshold")

#         mask = prediction['masks'][0, 0].mul(255).byte().cpu().numpy()
#         return mask

# def traditional_masking(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
#     kernel = np.ones((3, 3), np.uint8)
#     dilated_edges = cv2.dilate(edges, kernel, iterations=1)
#     mask = cv2.inRange(dilated_edges, 1, 255)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     return mask

# def mask_image_with_rcnn(image):
#     try:
#         mask = get_object_mask(image)
#     except ValueError:
#         mask = traditional_masking(image)
#     masked_image = cv2.bitwise_and(image, image, mask=mask)
#     return masked_image, mask

# def focus_stack(images):
#     stack_shape = images[0].shape[:2]
#     focus_measure = np.zeros(stack_shape)
#     focus_indices = np.zeros(stack_shape, dtype=int)

#     for i, image in enumerate(images):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         mask = laplacian > focus_measure
#         focus_measure[mask] = laplacian[mask]
#         focus_indices[mask] = i

#     stacked_image = np.zeros_like(images[0])
#     for y in range(stack_shape[0]):
#         for x in range(stack_shape[1]):
#             stacked_image[y, x] = images[focus_indices[y, x]][y, x]

#     return stacked_image, focus_indices

# def create_depth_map(focus_indices, layer_distance):
#     return focus_indices * layer_distance

# def depth_map_to_point_cloud(depth_map, image, xy_scale=1.0, z_scale=1.0):
#     h, w = depth_map.shape
#     points = []
#     colors = []
    
#     for y in range(h):
#         for x in range(w):
#             z = depth_map[y, x] * z_scale
#             if z != 0:
#                 points.append([x * xy_scale, y * xy_scale, z])
#                 color = image[y, x] / 255.0
#                 colors.append(color)

#     return np.array(points), np.array(colors)

# def calculate_dimensions(points):
#     if len(points) == 0:
#         return 0, 0, 0
#     x_min, y_min, z_min = np.min(points, axis=0)
#     x_max, y_max, z_max = np.max(points, axis=0)
#     return x_max - x_min, y_max - y_min, z_max - z_min

# # ----------------------- Metric Evaluation Functions ----------------------- #

# def evaluate_segmentation_metrics(image):
#     """
#     Evaluate segmentation quality by computing the area of the mask and the ratio of the masked area to the total image area.
#     """
#     _, mask = mask_image_with_rcnn(image)
#     area = np.sum(mask > 0)
#     total = mask.shape[0] * mask.shape[1]
#     ratio = area / total
#     return {"masked_area": int(area), "mask_ratio": ratio}

# def evaluate_focus_metrics(images_cleaned):
#     """
#     Evaluate the focus quality of each cleaned image by computing the average absolute Laplacian value.
#     """
#     focus_measures = []
#     for image in images_cleaned:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         focus_measures.append(np.mean(np.abs(laplacian)))
#     return {"mean_focus": np.mean(focus_measures), "std_focus": np.std(focus_measures)}

# def evaluate_point_cloud_metrics(point_cloud):
#     """
#     Evaluate the generated point cloud by counting the number of points and computing its bounding dimensions.
#     """
#     points = np.array(point_cloud)
#     if points.shape[0] == 0:
#         return {"num_points": 0, "bounding_box": (0, 0, 0)}
#     x_min, y_min, z_min = np.min(points, axis=0)
#     x_max, y_max, z_max = np.max(points, axis=0)
#     dims = (x_max - x_min, y_max - y_min, z_max - z_min)
#     return {"num_points": int(points.shape[0]), "bounding_box": dims}

# def evaluate_all_metrics(folder_path, layer_distance=100, xy_scale=0.01, z_scale=0.001):
#     """
#     Process images from the folder and evaluate metrics for each processing step:
#     - Segmentation metrics (masked area, mask ratio)
#     - Focus quality metrics
#     - Processing times for segmentation, focus stacking, and point cloud creation
#     - Point cloud metrics (number of points and bounding box)
#     - Calculated physical dimensions from the point cloud
#     """
#     start_total = time.time()
#     images = load_images_from_folder(folder_path)
    
#     seg_metrics_list = []
#     images_cleaned = []
#     segmentation_times = []
    
#     # Evaluate segmentation per image
#     for img in images:
#         start_seg = time.time()
#         masked_img, _ = mask_image_with_rcnn(img)
#         segmentation_times.append(time.time() - start_seg)
#         images_cleaned.append(masked_img)
#         seg_metrics_list.append(evaluate_segmentation_metrics(img))
        
#     avg_seg_time = np.mean(segmentation_times)
    
#     # Evaluate focus metrics on the cleaned images
#     focus_metrics = evaluate_focus_metrics(images_cleaned)
    
#     # Focus stacking
#     start_focus = time.time()
#     stacked_image, focus_indices = focus_stack(images_cleaned)
#     focus_stack_time = time.time() - start_focus
    
#     # Create depth map and generate point cloud
#     depth_map = create_depth_map(focus_indices, layer_distance)
#     start_pc = time.time()
#     point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale, z_scale)
#     pc_time = time.time() - start_pc
#     pc_metrics = evaluate_point_cloud_metrics(point_cloud)
    
#     # Calculate dimensions from the point cloud
#     dims = calculate_dimensions(point_cloud)
#     total_time = time.time() - start_total

#     metrics = {
#         "num_images": len(images),
#         "avg_segmentation_time_sec": avg_seg_time,
#         "segmentation_metrics_per_image": seg_metrics_list,
#         "focus_metrics": focus_metrics,
#         "focus_stack_time_sec": focus_stack_time,
#         "point_cloud_creation_time_sec": pc_time,
#         "point_cloud_metrics": pc_metrics,
#         "calculated_dimensions_mm": {
#             "length": dims[0],
#             "breadth": dims[1],
#             "height": dims[2]
#         },
#         "total_processing_time_sec": total_time,
#     }
#     return metrics

# # ----------------------- GUI Application (Optional) ----------------------- #

# class PointCloudApp:
#     def __init__(self, master):
#         self.master = master
#         self.master.title('3D Point Cloud Reconstruction Tool')
#         self.master.geometry('800x600')
#         self.frame = Frame(master)
#         self.frame.pack()
        
#         Label(self.frame, text="3D Point Cloud Reconstruction", font=("Arial", 18, "bold")).pack()
#         Button(self.frame, text='Upload Images', command=self.upload_images).pack(pady=10)
#         self.process_button = Button(self.frame, text='Process Images', command=self.start_processing)
#         self.process_button.pack(pady=10)
#         self.dimension_label = Label(self.frame, text="")
#         self.dimension_label.pack(pady=10)
#         self.metrics_label = Label(self.frame, text="", justify="left")
#         self.metrics_label.pack(pady=10)
#         self.progress_bar = ttk.Progressbar(self.frame, length=300, mode='determinate')
#         self.progress_bar.pack(pady=10)
#         self.images = []

#     def upload_images(self):
#         folder_path = filedialog.askdirectory()
#         if folder_path:
#             self.images = load_images_from_folder(folder_path)
#             messagebox.showinfo("Info", f"{len(self.images)} images loaded.")
    
#     def start_processing(self):
#         if not self.images:
#             messagebox.showwarning("Warning", "Please upload images first.")
#             return
#         threading.Thread(target=self.run_point_cloud_processing, daemon=True).start()
    
#     def run_point_cloud_processing(self):
#         self.progress_bar["maximum"] = len(self.images)
#         images_cleaned = []
        
#         for i, img in enumerate(self.images):
#             masked_img, _ = mask_image_with_rcnn(img)
#             images_cleaned.append(masked_img)
#             self.master.after(0, self.progress_bar.step, 1)
            
#         if not images_cleaned:
#             self.dimension_label.config(text="Error: No valid images after processing.")
#             return
        
#         stacked_image, focus_indices = focus_stack(images_cleaned)
#         depth_map = create_depth_map(focus_indices, layer_distance=100)
#         point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale=0.01, z_scale=0.001)
#         length, breadth, height = calculate_dimensions(point_cloud)
        
#         self.master.after(0, self.dimension_label.config, {"text": f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm'})
#         self.visualize_point_cloud(point_cloud, colors)

#         # Optionally, compute and display metrics for this run
#         metrics = evaluate_all_metrics_from_images(self.images)
#         metrics_text = json.dumps(metrics, indent=4)
#         self.master.after(0, self.metrics_label.config, {"text": metrics_text})

#     def visualize_point_cloud(self, points, colors):
#         if len(points) == 0:
#             return
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors)
#         o3d.visualization.draw_geometries([pcd])

# def evaluate_all_metrics_from_images(images, layer_distance=100, xy_scale=0.01, z_scale=0.001):
#     """
#     A helper function similar to evaluate_all_metrics() but using a list of images (already loaded).
#     """
#     start_total = time.time()
    
#     seg_metrics_list = []
#     images_cleaned = []
#     segmentation_times = []
    
#     for img in images:
#         start_seg = time.time()
#         masked_img, _ = mask_image_with_rcnn(img)
#         segmentation_times.append(time.time() - start_seg)
#         images_cleaned.append(masked_img)
#         seg_metrics_list.append(evaluate_segmentation_metrics(img))
        
#     avg_seg_time = np.mean(segmentation_times)
#     focus_metrics = evaluate_focus_metrics(images_cleaned)
    
#     start_focus = time.time()
#     stacked_image, focus_indices = focus_stack(images_cleaned)
#     focus_stack_time = time.time() - start_focus
    
#     depth_map = create_depth_map(focus_indices, layer_distance)
#     start_pc = time.time()
#     point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale, z_scale)
#     pc_time = time.time() - start_pc
#     pc_metrics = evaluate_point_cloud_metrics(point_cloud)
#     dims = calculate_dimensions(point_cloud)
#     total_time = time.time() - start_total

#     metrics = {
#         "num_images": len(images),
#         "avg_segmentation_time_sec": avg_seg_time,
#         "segmentation_metrics_per_image": seg_metrics_list,
#         "focus_metrics": focus_metrics,
#         "focus_stack_time_sec": focus_stack_time,
#         "point_cloud_creation_time_sec": pc_time,
#         "point_cloud_metrics": pc_metrics,
#         "calculated_dimensions_mm": {
#             "length": dims[0],
#             "breadth": dims[1],
#             "height": dims[2]
#         },
#         "total_processing_time_sec": total_time,
#     }
#     return metrics

# # ----------------------- Command-Line Evaluation Script ----------------------- #

# if __name__ == "__main__":
#     # When running from the command line, you can choose between the GUI or metrics evaluation mode.
#     import argparse
#     parser = argparse.ArgumentParser(description="3D Point Cloud Reconstruction Metrics Evaluation")
#     parser.add_argument("--folder", type=str, help="Path to folder with images")
#     parser.add_argument("--gui", action="store_true", help="Launch the GUI application")
#     args = parser.parse_args()

#     if args.gui:
#         root = Tk()
#         app = PointCloudApp(root)
#         root.mainloop()
#     elif args.folder:
#         metrics = evaluate_all_metrics(args.folder)
#         print(json.dumps(metrics, indent=4))
#     else:
#         print("Please provide either a folder path with --folder or launch the GUI with --gui")

import sys
import cv2
import numpy as np
import glob
import torch
import threading
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame
from tkinter import ttk
import open3d as o3d
import os
import gc
from sklearn.metrics import mean_squared_error

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

def depth_smoothness(depth_map):
    depth_map = depth_map.astype(np.float32)  # Ensure it's in a supported format
    laplacian = cv2.Laplacian(depth_map, cv2.CV_64F)
    return np.var(laplacian)


def rmse_depth_map(depth_map, gt_depth_map):
    return np.sqrt(mean_squared_error(depth_map.flatten(), gt_depth_map.flatten()))

def point_cloud_density(num_points, bounding_box):
    volume = bounding_box[0] * bounding_box[1] * bounding_box[2]
    return num_points / volume

def compute_surface_roughness(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    normals = np.asarray(pcd.normals)
    return np.var(normals)

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

class PointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title('3D Point Cloud Reconstruction Tool')
        self.master.geometry('800x600')
        self.frame = Frame(master)
        self.frame.pack()
        
        Label(self.frame, text="3D Point Cloud Reconstruction", font=("Arial", 18, "bold")).pack()
        Button(self.frame, text='Upload Images', command=self.upload_images).pack(pady=10)
        self.process_button = Button(self.frame, text='Process Images', command=self.start_processing)
        self.process_button.pack(pady=10)
        self.dimension_label = Label(self.frame, text="")
        self.dimension_label.pack(pady=10)
        self.progress_bar = ttk.Progressbar(self.frame, length=300, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.images = []

    def upload_images(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.images = load_images_from_folder(folder_path)
            messagebox.showinfo("Info", f"{len(self.images)} images loaded.")
    
    def start_processing(self):
        if not self.images:
            messagebox.showwarning("Warning", "Please upload images first.")
            return
        threading.Thread(target=self.run_point_cloud_processing, daemon=True).start()
    
    def run_point_cloud_processing(self):
        self.progress_bar["maximum"] = len(self.images)
        images_cleaned = []
        
        for i, img in enumerate(self.images):
            masked_img, _ = mask_image_with_rcnn(img)
            images_cleaned.append(masked_img)
            self.master.after(0, self.progress_bar.step, 1)
            
        stacked_image, focus_indices = focus_stack(images_cleaned)
        depth_map = create_depth_map(focus_indices, layer_distance=100)
        point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale=0.01, z_scale=0.001)
        length, breadth, height = calculate_dimensions(point_cloud)
        density = point_cloud_density(len(point_cloud), (length, breadth, height))
        smoothness = depth_smoothness(depth_map)
        roughness = compute_surface_roughness(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud)))
        
        self.master.after(0, self.dimension_label.config, {
            "text": f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm, '
                    f'Density: {density:.2f}, Smoothness: {smoothness:.2f}, Roughness: {roughness:.2f}'
        })

if __name__ == "__main__":
    root = Tk()
    app = PointCloudApp(root)
    root.mainloop()
