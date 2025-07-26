import open3d as o3d
import numpy as np
import os
from PIL import Image
import torch

import matplotlib.pyplot as plt


def create_colormap_colors(index_array):
    """
    Convert token indexes into colors using a red-to-green colormap.
    """
    norm_idx = (index_array - index_array.min()) / (
        index_array.max() - index_array.min()
    )  # Normalize [0,1]
    cmap = plt.get_cmap("turbo")  # Red to Green colormap
    colors = cmap(norm_idx)[:, :3]  # Extract only RGB channels
    return colors


def create_sphere(center, radius=0.0, color=[0, 1, 0]):
    """Creates a high-resolution sphere mesh at a given center."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()  # Enable shading
    sphere.translate(center)
    return sphere


def set_camera(view_ctl, azimuth, elevation, distance=1.5):
    """Set camera position based on azimuth and elevation angles (degrees)."""
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)

    # Convert spherical coordinates to Cartesian
    cam_x = distance * np.cos(elevation) * np.cos(azimuth)
    cam_y = distance * np.cos(elevation) * np.sin(azimuth)
    cam_z = distance * np.sin(elevation)

    # Set the camera position
    view_ctl.set_front([-cam_x, -cam_y, -cam_z])  # Direction to look at the origin
    view_ctl.set_lookat([0, 0, 0])  # Always look at the origin
    view_ctl.set_up([0, 1, 0])  # Up direction
    view_ctl.set_zoom(0.8)  # Adjust zoom level


def assign_first_occurrence_index(points, index_array):
    """
    Finds repeated points and assigns them the index of their first occurrence.

    Parameters:
        points (numpy.ndarray): (n,3) array of point cloud coordinates.
        index_array (numpy.ndarray): (n,) array of indexes associated with each point.

    Returns:
        numpy.ndarray: Updated index array with repeated points assigned to the first occurrence index.
    """
    # Find unique points and their first occurrences
    unique_points, first_occurrence_indices = np.unique(
        points, axis=0, return_index=True
    )

    # Create a mapping from point tuple to first occurrence index
    point_to_index = {
        tuple(points[i]): index_array[
            first_occurrence_indices[np.where(unique_points == points[i])[0][0]]
        ]
        for i in range(len(points))
    }

    # Assign the index of the first occurrence to all duplicates
    updated_index_array = np.array([point_to_index[tuple(p)] for p in points])

    return updated_index_array


dir = "lab/visualize_purge/corrupted_plane_background/"

with_cls = "" # "_cls"

centers = torch.load(dir + "center.pth", weights_only=True)[0]
neighbors = torch.load(dir + "neighborhood.pth", weights_only=True)[0]
sorted_indexes = torch.load(dir + f"sorted_index{with_cls}.pth", weights_only=True)
sorted_indexes= torch.argsort(sorted_indexes, dim=0)
sorted_indexes = sorted_indexes.unsqueeze(1).repeat(1, neighbors.size(1))

neighbors = neighbors.reshape(-1, 3)
sorted_indexes = sorted_indexes.reshape(-1)


centers = centers.numpy()
neighbors = neighbors.numpy()
sorted_indexes = sorted_indexes.numpy()

sorted_indexes = assign_first_occurrence_index(neighbors, sorted_indexes)


colors = create_colormap_colors(sorted_indexes)

# Create a list of spheres
spheres = [create_sphere(point, radius=0.03, color=[0.8, 0.4, 0.1]) for point in neighbors]

# spheres = [
#     create_sphere(point, radius=0.03, color=colors[i])
#     for i, point in enumerate(neighbors)
# ]

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)  # Off-screen rendering

# Add all spheres
for sphere in spheres:
    vis.add_geometry(sphere)

# Set render options
opt = vis.get_render_option()
opt.background_color = np.array([1, 1, 1])  # White background (for transparency)
opt.light_on = True  # Enable lighting

# Get view control
view_ctl = vis.get_view_control()
set_camera(
    view_ctl, azimuth=-30, elevation=30
)  # Change these values for different angles

# Render the scene and capture the image
vis.poll_events()
vis.update_renderer()
image_path = "temp_render.png"
vis.capture_screen_image(image_path, do_render=True)  # Save PNG
vis.destroy_window()

print("Saved temporary image as:", image_path)

# **Post-process to remove background (Make Transparent)**
img = Image.open(image_path).convert("RGBA")
data = np.array(img)

# Replace white background with transparency
r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
white_mask = (r > 240) & (g > 240) & (b > 240)  # Detect white background
data[white_mask] = [255, 255, 255, 0]  # Make white pixels transparent

# Save as transparent PNG
transparent_img = Image.fromarray(data)

transparent_img.save(os.path.join(dir, f"corrupted_{with_cls}.png"))

print("Final image saved as 'point_cloud_transparent.png' with transparency!")
