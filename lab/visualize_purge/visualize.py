import open3d as o3d
import numpy as np
import os
from PIL import Image


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
    

# file_path = "lab/visualize_purge/original_plane/pointcloud_0.ply"
file_path = "lab/visualize_purge/original_plane/pointcloud_0.ply"


point_cloud = o3d.io.read_point_cloud(file_path)
# Convert to NumPy array
points = np.asarray(point_cloud.points)

# Create a list of spheres
spheres = [create_sphere(point, radius=0.03, color=[0, 0.35, 0]) for point in points]

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
set_camera(view_ctl, azimuth=-30, elevation=30)  # Change these values for different angles

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

dir = os.path.dirname(file_path)
filename = os.path.splitext(os.path.basename(file_path))[0]
transparent_img.save(os.path.join(dir, f"{filename}.png"))

print("Final image saved as 'point_cloud_transparent.png' with transparency!")
