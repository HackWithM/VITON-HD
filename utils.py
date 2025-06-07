import torch
import torchvision.utils as vutils
import os

def gen_noise(shape, device='cpu'):
    """
    Generates a tensor of random noise.
    Args:
        shape (tuple): The desired shape of the noise tensor (e.g., (batch_size, channels, height, width)).
        device (str): The device to create the tensor on ('cpu' or 'cuda').
    Returns:
        torch.Tensor: A tensor of random noise.
    """
    return torch.randn(shape, device=device)

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Loads a model checkpoint.
    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state for. Defaults to None.
        device (str): The device to load the checkpoint to ('cpu' or 'cuda').
    Returns:
        int: The epoch number from the checkpoint, or 0 if not found.
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found: {filepath}")
        return 0
    try:
        # FutureWarning: You are using `torch.load` with `weights_only=False`...
        # We will set weights_only=True for security and future compatibility.
        # If this causes issues, it might be because the checkpoint stores more than just weights
        # and those other objects are not in the allowlist.
        checkpoint = torch.load(filepath, map_location=device, weights_only=True) # Added weights_only=True

        try:
            # Standard case: checkpoint is a dict with 'model_state_dict'
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            print(f"Checkpoint loaded (standard structure) from {filepath} at epoch {epoch}")
        except TypeError: # This can happen if checkpoint is not a dict (e.g. if weights_only=True returns only tensors)
            print(f"Checkpoint from {filepath} is not a dictionary (likely due to weights_only=True). Assuming it's the model state_dict itself.")
            model.load_state_dict(checkpoint)
            epoch = 0 # Epoch info and optimizer state likely not available in this case
            if optimizer:
                print(f"Optimizer state not loaded from {filepath} as checkpoint is assumed to be state_dict only.")
            print(f"Checkpoint loaded (assumed state_dict directly) from {filepath}")
        except KeyError:
            # Fallback: checkpoint itself is the state_dict, or 'model_state_dict' key is missing
            print(f"Key 'model_state_dict' not found in checkpoint from {filepath}. Assuming checkpoint is the state_dict itself or has a different structure.")
            model.load_state_dict(checkpoint) # Try loading the whole checkpoint object as state_dict
            epoch = 0 # Epoch info likely not available
            # Attempt to load optimizer state if checkpoint is a dict and has optimizer_state_dict
            # This is less likely if 'model_state_dict' was missing but checkpoint was a dict.
            if optimizer and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Optimizer state loaded from {filepath} despite unusual model state_dict structure.")
            elif optimizer:
                print(f"Optimizer state not loaded from {filepath} as checkpoint structure is unexpected for optimizer.")
            print(f"Checkpoint loaded (fallback structure) from {filepath}")

        return epoch
    except Exception as e:
        print(f"Error loading checkpoint from {filepath}: {e}")
        # If weights_only=True causes issues with non-tensor objects, the error might originate here.
        # For example, if the checkpoint contains custom classes not allowed by weights_only=True.
        return 0

def save_images(images, filenames, directory, nrow=8):
    """
    Saves a batch of images, each to its own file.
    Args:
        images (torch.Tensor): A batch of images (e.g., (batch_size, channels, height, width)).
        filenames (list of str): A list of filenames for each image in the batch.
        directory (str): The directory to save the images in.
        nrow (int): Number of images to display in each row if saving as a grid (unused if saving individual files).
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if images.size(0) != len(filenames):
        print(f"Error: Number of images ({images.size(0)}) does not match number of filenames ({len(filenames)}).")
        # Optionally, save as a grid to a default name if counts mismatch
        grid_path = os.path.join(directory, "image_grid_error.png")
        try:
            vutils.save_image(images, grid_path, nrow=nrow, normalize=True)
            print(f"Saved image grid to {grid_path} due to filename mismatch.")
        except Exception as e_grid:
            print(f"Error saving image grid to {grid_path}: {e_grid}")
        return

    for i, image_tensor in enumerate(images):
        # image_tensor is (C, H, W), vutils.save_image expects (C, H, W) or (B, C, H, W)
        # If saving single image, it's fine.
        file_path = os.path.join(directory, filenames[i])
        try:
            # Ensure the image tensor is 3D (C, H, W) or 4D with batch_size=1 for save_image
            # If image_tensor is already (C,H,W) from iterating batch, it's fine.
            # If it's (1,C,H,W) after unbind/split, also fine.
            # If it's part of a batch, ensure it's correctly sliced.
            vutils.save_image(image_tensor, file_path, normalize=True) # nrow is not used for single image
            print(f"Image saved to {file_path}")
        except Exception as e:
            print(f"Error saving image to {file_path}: {e}")

# Example usage (optional, for testing this module directly)
if __name__ == '__main__':
    # Test gen_noise
    noise = gen_noise((1, 3, 64, 64))
    print("Generated noise tensor shape:", noise.shape)

    # Test save_images (creates a dummy image)
    # Create a directory for test outputs if it doesn't exist
    if not os.path.exists('test_outputs'):
        os.makedirs('test_outputs')
    dummy_images = torch.rand((4, 3, 32, 32)) # 4 random images
    save_images(dummy_images, 'test_outputs/dummy_grid.png', nrow=2)

    # Test load_checkpoint (requires a dummy model and checkpoint)
    # This part is more complex to demonstrate without an actual model and saved checkpoint.
    # You would typically have:
    # model = YourModelClass()
    # optimizer = torch.optim.Adam(model.parameters())
    # torch.save({
    #     'epoch': 10,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, 'test_outputs/dummy_checkpoint.pth')
    # epoch_loaded = load_checkpoint('test_outputs/dummy_checkpoint.pth', model, optimizer)
    # print(f"Loaded checkpoint from epoch: {epoch_loaded}")
    print("Utils module functions defined.")