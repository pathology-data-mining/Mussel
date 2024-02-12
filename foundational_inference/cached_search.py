from utils import load_quilt
from torch.utils.data import Dataset
from torch.nn import CosineSimilarity
import os
import torch
import glob
import h5py
import openslide
import matplotlib.pyplot as plt
from datetime import datetime
import heapq
import numpy as np


class CachedImageEmbeddingDataset(Dataset):
    def __init__(self, _quilt, _img_emb_dir, n_tiles, prototype=False):
        self.txt_tokenizer = _quilt['tokenizer']
        self.data_paths = glob.glob(os.path.join(_img_emb_dir, "*.pt"))
        if prototype:
            self.data_paths = self.data_paths[:10]
        self.image_ids = [os.path.basename(_path).replace(".pt", "") for _path in self.data_paths]
        self.n_tiles = n_tiles

    def __len__(self):
        return len(self.data_paths)
    
    def _pad(self, _emb):
        n, d = _emb.shape
        if n < self.n_tiles:
            emb = torch.cat((_emb, torch.zeros(self.n_tiles - n, d)))
            indices = np.concatenate((np.arange(n), np.zeros(self.n_tiles - n)))
        elif n > self.n_tiles:
            indices = np.random.choice(n, self.n_tiles, replace=False)
            emb = _emb[indices]
        else:
            emb = _emb
            indices = np.arange(n)
        return emb, indices

    def __getitem__(self, _idx):
        _emb_path = self.data_paths[_idx]
        _emb = torch.load(_emb_path)
        _emb, indices = self._pad(_emb)
        _image_id = self.image_ids[_idx]
        indices = indices.astype(int)

        return {'image_id': _image_id, 'emb': _emb, "indices": indices}


class QueryEmbedder(torch.nn.Module):
    def __init__(self, _quilt, device):
        super(QueryEmbedder, self).__init__()
        self.quilt_model = _quilt['model']
        self.quilt_tokenizer = _quilt['tokenizer']
        self.device = device
        self.quilt_model.to(self.device)

    def forward(self, text):
        tokens = self.quilt_tokenizer(text)
        with torch.no_grad():
            return self.quilt_model.encode_text(tokens.to(self.device))


class CachedSearch():
    def __init__(self, _quilt, _img_emb_dir, _patch_dir, _slide_dir, _output_dir, _device_str, prototype=False):
        self.patch_dir = _patch_dir
        self.slide_dir = _slide_dir
        self.output_dir = _output_dir
        self.n_tiles_per_slide = 10
        self.img_emb_dataset = CachedImageEmbeddingDataset(_quilt, _img_emb_dir, self.n_tiles_per_slide, prototype=prototype)
        self.len = len(self.img_emb_dataset)
        self.img_emb_loader = torch.utils.data.DataLoader(self.img_emb_dataset,
                                                          batch_size=256,
                                                          shuffle=False,
                                                          num_workers=64,
                                                          pin_memory=True,)
        self.device = torch.device(_device_str)
        self.query_embedder = QueryEmbedder(_quilt, self.device)
        self.cos_sim = CosineSimilarity(dim=2, eps=1e-6)

    def search(self, _query, _top_k=4):
        query_emb = self.query_embedder(_query).to(self.device)
        top_k_results = []  # min-heap for top k results

        for batch in self.img_emb_loader:
            emb = batch['emb'].to(self.device)
            sim = self.cos_sim(query_emb, emb)
            maxes_by_slide, argmax_tile = sim.max(dim=1)  # max similarity for each slide, indexed within sample of n_tiles_per_slide
            max_, argmax_slide = maxes_by_slide.max(dim=0)  # max similarity overall

            # push the result into the heap if the heap is not full
            # or the new result is larger than the smallest result in the heap
            if len(top_k_results) < _top_k or max_ > top_k_results[0][0]:
                if len(top_k_results) == _top_k:
                    heapq.heappop(top_k_results)
                argmax_tile_in_batch = batch['indices'][argmax_slide][argmax_tile[argmax_slide]]
                heapq.heappush(top_k_results,
                               (max_,
                                batch['image_id'][argmax_slide],
                                argmax_tile_in_batch))

        # display the top k images
        image_ids, tile_indices = [], []
        while top_k_results:
            max_, image_id, tile_index = heapq.heappop(top_k_results)
            print(f"'{_query}' is similar to {image_id} at index {tile_index} with similarity {max_:.3f}")
            image_ids.append(image_id)
            tile_indices.append(tile_index)
        return image_ids, tile_indices
        
    def load_image(self, image_id, tile_index):
        patch_file = os.path.join(self.patch_dir, f"{image_id}.h5")
        with h5py.File(patch_file, "r") as f:
            patch_size = f["coords"].attrs["patch_size"]
            patch_level = f["coords"].attrs["patch_level"]
            coord = f["coords"][tile_index]

        slide_list = pd.read_csv("/gpfs/mskmind_ess/pdm/reef/slide_directory.csv")
        try:
            slide_file = slide_list[slide_list['image_id'].astype(str) == str(image_id)]['slide_path'].values[0]
            wsi = openslide.OpenSlide(slide_file)
        except:
            print(f"Could not open {slide_file}")
            return None
        img = wsi.read_region(coord, patch_level, (patch_size, patch_size))
        return img
            
    def save_images(self, image_ids, tile_indices, _query, search_timestamp):
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))

        for i, (_image_id, _tile_index) in enumerate(zip(image_ids, tile_indices)):
            img = self.load_image(_image_id, _tile_index)

            ax = axs[i // 2, i % 2]
            ax.imshow(img)
            ax.axis('off')

        # place caption below image with query
        fig.text(0.5, 0.04, _query, ha='center', va='center', fontsize=12, wrap=True, bbox=dict(facecolor='white', alpha=0.5))

        plt.savefig(os.path.join(self.output_dir, f"{search_timestamp}_top4.png"))
        plt.close()

if __name__ == "__main__":
    print("******************************************************\n"
          "*        Welcome to Memorial Sloan Kettering's       *\n"
          "*         Pathology Data Mining image search!        *\n"
          "******************************************************\n\n"
          "Loading foundation model; please wait....\n")

    QUILT = load_quilt()
    IMG_EMB_DIR = "/gpfs/mskmind_ess/pdm/reef/1.0_224_896_None/feats/quilt/pt"
    PATCH_DIR = "/gpfs/mskmind_ess/pdm/reef/1.0_224_896_None/patches"
    SLIDE_DIR = "/gpfs/mskmind_emc/data_large/pathology/BR_20-226/slides"
    OUTPUT_DIR = "/gpfs/mskmind_ess/boehmk/scratch/queries"

    CS = CachedSearch(QUILT, IMG_EMB_DIR, PATCH_DIR, SLIDE_DIR, OUTPUT_DIR, "cpu", prototype=True)
    print("Thank you for waiting. Ready for your query. Type 'exit' to close.")

    PROMPT_STRING = "Please enter query: "
    QUERY = input(PROMPT_STRING)
    while QUERY != "exit":
        IMAGE_IDS, TILE_INDICES = CS.search(QUERY)
        SEARCH_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        CS.save_images(IMAGE_IDS, TILE_INDICES, QUERY, SEARCH_TIMESTAMP)

        QUERY = input(PROMPT_STRING)

    print("Thank you for using the Pathology Data Mining team's image search. Have a great day.")
    # cs.search("invasive ductal carcinoma with infiltrating lymphocytes")
    # cs.search("confluent round blue cells with conspicuous nucleoli and pleomorphic nuclei")
    # cs.search("artifact, specifically marking pen")
