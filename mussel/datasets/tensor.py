from typing import List

import torch
from torch.utils.data import Dataset
from enum import Enum

class SiteType(Enum):
    PRIMARY = "Primary"
    METASTASIS = "Metastasis"


class TileFeatureTensorDataset(Dataset):
    def __init__(
        self,
        site_type: SiteType,
        tile_feature_tensor_path: str,
        n_max_tiles: int = 20000,
    ) -> None:
        """Initialize the dataset.

        Args:
            site_type: the site type as str, either "Primary" or "Metastasis"
            tile_tensor_path: the path of the tile tensor as str
            n_max_tiles: the maximum number of tiles to use as int

        Returns:
            None
        """
        self.site_type = site_type
        self.tile_feature_tensor_path = tile_feature_tensor_path
        self.n_max_tiles = n_max_tiles

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: the length of the dataset
        """
        return len(self.sample_ids)

    def get_tile_tensor(self, tile_tensor_path) -> torch.Tensor:
        """Get the tile tensor.

        Args:
            tile_tensor_path: the path of the tile tensor as str or list of str

        Returns:
            torch.Tensor: the tile tensor
        """
        path_or_paths = tile_tensor_path.split("|||")
        tile_tensor = []
        for tile_path in path_or_paths:
            emb = torch.load(tile_path, weights_only=True)
            # if emb.shape[0] > self.n_max_tiles:
            #     indices = torch.randperm(emb.shape[0])[: self.n_max_tiles]
            #     emb = emb[indices]
            tile_tensor.append(emb)
        tile_tensor = torch.cat(tile_tensor, dim=0)
        if tile_tensor.shape[0] > self.n_max_tiles:
            indices = torch.randperm(tile_tensor.shape[0])[: self.n_max_tiles]
            tile_tensor = tile_tensor[indices]
        if tile_tensor.shape[0] < self.n_max_tiles:
            padding = torch.zeros(
                self.n_max_tiles - tile_tensor.shape[0], tile_tensor.shape[1]
            )
            tile_tensor = torch.cat([tile_tensor, padding], dim=0)
        return tile_tensor

    def __getitem__(self, idx: int) -> dict:
        """Return an item from the dataset.

        Args:
            idx: the index of the item to return

        Returns:
            dict: the item
        """
        tile_tensor = self.get_tile_tensor(self.tile_feature_tensor_path)
        return {
            "site": self.site_type,
            "tile_tensor": tile_tensor,
        }
