import torch.nn as nn
import numpy as np
from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer, TileMerger
from torch.utils.data import Dataset, DataLoader

def predict(model: nn.Module, image: np.ndarray, image_size, normalize=A.Normalize(), batch_size=1) -> np.ndarray:

    tile_step = (image_size[0] // 2, image_size[1] // 2)

    tile_slicer = ImageSlicer(image.shape, image_size, tile_step)
    tile_merger = TileMerger(tile_slicer.target_shape,
                             1, tile_slicer.weight, device="cuda")
    patches = tile_slicer.split(image)

    transform = A.Compose([normalize, A.Lambda(image=_tensor_from_rgb_image)])

    data = list(
        {"image": patch, "coords": np.array(coords, dtype=np.int)}
        for (patch, coords) in zip(patches, tile_slicer.crops)
    )
    for batch in DataLoader(
        InMemoryDataset(data, transform),
        pin_memory=True,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    ):
        image = batch["image"].cuda(non_blocking=True)
        coords = batch["coords"]
        output = model(image)
        tile_merger.integrate_batch(output, coords)

    mask = tile_merger.merge()

    mask = np.moveaxis(to_numpy(mask), 0, -1)
    mask = tile_slicer.crop_to_orignal_size(mask)

    return mask


if __name__ == "__main__":
    predict()
