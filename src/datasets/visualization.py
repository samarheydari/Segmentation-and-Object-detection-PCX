

def visualize_objects(image, objects, class_id_to_label_fn=None, padding=0.3, figsize_full=(12, 12), 
                      figsize_obj=(5, 5), max_cols=3, show_full=True):
    """
    Visualize objects with bounding boxes in an image.
    Objects are same as returned by datasets.person_car_dataset.PersonCarDataset
    
    Parameters:
    -----------
    image : torch.Tensor or numpy.ndarray
        The image tensor (C, H, W) or array (H, W, C)
    objects : list of dict
        List of object dictionaries with keys: x_center, y_center, width, height, class_id
    class_id_to_label_fn : callable, optional
        Function to convert class_id to label string. If None, class_id is used as label.
    padding : float, optional
        Padding around objects for zoomed views, as a fraction of object size
    figsize_full : tuple, optional
        Figure size for the full image plot
    figsize_obj : tuple, optional
        Base figure size for each object subplot (will be multiplied by number of columns)
    max_cols : int, optional
        Maximum number of columns in the grid of object subplots
    show_full : bool, optional
        Whether to show the full image with all bounding boxes
        
    Returns:
    --------
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert image to numpy array if it's a tensor
    if hasattr(image, 'permute') and hasattr(image, 'numpy'):
        image_np = image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    elif isinstance(image, np.ndarray):
        if image.shape[0] == 3 and len(image.shape) == 3:  # If (C, H, W)
            image_np = np.transpose(image, (1, 2, 0))
        else:
            image_np = image  # Assume already in (H, W, C) format
    else:
        raise TypeError("Image must be a torch.Tensor or numpy.ndarray")
    
    # Get image dimensions
    image_height, image_width = image_np.shape[:2]
    
    # Define class_id to label conversion function if not provided
    if class_id_to_label_fn is None:
        class_id_to_label_fn = lambda x: f"Class {x}"
    
    # Show the full image with all bounding boxes
    if show_full:
        fig, ax = plt.subplots(1, 1, figsize=figsize_full)
        ax.imshow(image_np)
        
        for i, object in enumerate(objects):
            x_center = object["x_center"]
            y_center = object["y_center"]
            width = object["width"]
            height = object["height"]
            class_id = object["class_id"]
            
            # Transform from normalized coordinates to image coordinates
            x0 = (x_center - width / 2) * image_width
            y0 = (y_center - height / 2) * image_height
            width_px = width * image_width
            height_px = height * image_height
            
            ax.add_patch(plt.Rectangle((x0, y0), width_px, height_px, linewidth=2, edgecolor="y", facecolor="none"))
            class_label = class_id_to_label_fn(class_id) 
            ax.text(x0, y0, f"{i}: {class_label}", fontsize=12, color="y")
            
        plt.title("Full Image with Bounding Boxes")
        plt.tight_layout()
        plt.show()
    
    # Create a zoomed plot for each object
    if len(objects) > 0:
        n_cols = min(max_cols, len(objects))  # Maximum max_cols objects per row
        n_rows = (len(objects) + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_obj[0]*n_cols, figsize_obj[1]*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, object in enumerate(objects):
            x_center = object["x_center"]
            y_center = object["y_center"]
            width = object["width"]
            height = object["height"]
            class_id = object["class_id"]
            class_label = class_id_to_label_fn(class_id)
            
            # Transform from normalized coordinates to image coordinates
            x0 = (x_center - width / 2) * image_width
            y0 = (y_center - height / 2) * image_height
            width_px = width * image_width
            height_px = height * image_height
            
            # Calculate zoomed region (add padding around object)
            zoom_x0 = max(0, x0 - padding * width_px)
            zoom_y0 = max(0, y0 - padding * height_px)
            zoom_width = min(image_width - zoom_x0, width_px * (1 + 2*padding))
            zoom_height = min(image_height - zoom_y0, height_px * (1 + 2*padding))
            
            # Create zoomed image
            axes[i].imshow(image_np)
            axes[i].set_xlim(zoom_x0, zoom_x0 + zoom_width)
            axes[i].set_ylim(zoom_y0 + zoom_height, zoom_y0)  # Reverse y-axis for proper display
            
            # Add bounding box
            axes[i].add_patch(plt.Rectangle((x0, y0), width_px, height_px, linewidth=2, edgecolor="r", facecolor="none"))
            axes[i].text(x0, y0, f"{class_label}", fontsize=12, color="r", backgroundcolor="white")
            axes[i].set_title(f"Object {i}: {class_label}")
            
        # Hide any unused subplots
        for j in range(len(objects), len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()
    else:
        print("No objects found in this image.")

# Example usage:
"""
# Define transformation (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
])
# Load dataset
root_dir = "../datasets/data/Arthal/"
train_dataset = PersonCarDataset(root_dir=root_dir, split="train", transform=transform)
img_idx = 7
image, objects = train_dataset[img_idx]

# Call the visualization function
visualize_objects(
    image=image, 
    objects=objects, 
    class_id_to_label_fn=train_dataset.class_id_to_label
)
"""