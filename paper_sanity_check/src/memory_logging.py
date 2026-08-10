import torch


def log_cuda_memory(logger, tag=""):
    """Log CUDA memory usage in a human-readable format"""
    if not torch.cuda.is_available():
        logger.info(f"{tag} CUDA not available")
        return
    
    # Get memory information
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
    
    # Get device properties
    device_props = torch.cuda.get_device_properties(0)
    total_memory = device_props.total_memory / (1024 ** 3)  # GB
    
    # Log memory usage
    logger.info(f"{tag} CUDA Memory: {allocated:.2f}GB allocated, {max_allocated:.2f}GB max allocated, "
                f"{reserved:.2f}GB reserved, {total_memory:.2f}GB total")