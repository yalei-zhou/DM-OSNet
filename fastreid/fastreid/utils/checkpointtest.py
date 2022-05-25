import torch
def load_file( f: str):
    """
    Load a checkpoint file. Can be overwritten by subclasses to support
    different formats.

    Args:
        f (str): a locally mounted file path.
    Returns:
        dict: with keys "model" and optionally others that are saved by
            the checkpointer dict["model"] must be a dict which maps strings
            to torch.Tensor or numpy arrays.
    """
    return torch.load(f, map_location=torch.device("cpu"))


ss = load_file("/home/zyl/fast-reid/checkpoints/1/per-training/lup_moco_r50.pth")
print(ss)