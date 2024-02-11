import open_clip
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import CosineSimilarity
from argparse import ArgumentParser


def load_quilt():
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-16-PMB')
    tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-16-PMB')
    return {'model': model, 'preprocess_train': preprocess_train, 'preprocess_val': preprocess_val,
            'tokenizer': tokenizer}


def get_toy_data_paths(k=50):
    img_path = "/gpfs/mskmind_ess/pdm/example-img.png"
    txt_path = "/gpfs/mskmind_ess/pdm/example-text.txt"
    record = {'image': img_path, 'text': txt_path, 'label': 0}
    return [record for _ in range(k)]


def load_example_pair():
    paths = get_toy_data_paths(k=1)
    img_path, txt_path = paths[0]['image'], paths[0]['text']
    image = Image.open(img_path)
    with open(txt_path, "r") as f:
        text = f.read()
    return {'image': image, 'text': text}


class ZeroShotDataset(Dataset):
    def __init__(self, _quilt, _data_paths):
        self.img_preprocess = _quilt['preprocess_val']
        self.txt_tokenizer = _quilt['tokenizer']
        self.data_paths = _data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, _idx):
        _image_path = self.data_paths[_idx]['image']
        _image = Image.open(_image_path)
        _image = self.img_preprocess(_image)

        _text_path = self.data_paths[_idx]['text']
        with open(_text_path, "r") as f:
            _text = f.read()
        _tokenized_text = self.txt_tokenizer(_text)

        _label = self.data_paths[_idx]['label']

        return {'image': _image, 'text': _tokenized_text, 'label': _label}

class ZeroShotClassifier(nn.Module):
    def __init__(self, _quilt, _classes):
        super(ZeroShotClassifier, self).__init__()
        self.model = _quilt['model']
        self.cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6)

        self.original_class_text = _classes
        tokenized_text = _quilt['tokenizer'](_classes)
        with torch.no_grad():
            self.register_buffer('class_embeddings', self.model.encode_text(tokenized_text))
        print(f"Initialized ZeroShotClassifier with {self.class_embeddings.shape[0]} classes")


    def similarity(self, _image_features, _text_features):
        # _image_features ~ batch, dim
        # _text_features ~ n_class, dim

        # convert to batch, n_class, dim
        _image_features = _image_features.unsqueeze(1).repeat(1, len(_text_features), 1)
        _text_features = _text_features.unsqueeze(0).repeat(_image_features.shape[0], 1, 1)

        cos_sim = self.cos_sim(_image_features, _text_features)
        return cos_sim

    def forward(self, image):
        with torch.no_grad():
            _image_features = self.model.encode_image(image)
        dist = self.similarity(_image_features, self.class_embeddings)

        # convert to logits (smaller values yield higher probability)
        logits = torch.softmax(-dist, dim=1)
        pred = torch.argmax(logits, dim=1)
        pred_class = [self.original_class_text[_pred] for _pred in pred]
        return {'logits': logits, 'pred': pred, 'pred_class': pred_class}


def main_simple():
    """
    test embedding
    :return: None
    """
    quilt = load_quilt()
    data = load_example_pair()

    # preprocess data
    image = quilt['preprocess_val'](data['image']).unsqueeze(0)
    text = quilt['tokenizer'](data['text'])

    # inference
    with torch.no_grad():
        image_features = quilt['model'].encode_image(image)
        text_features = quilt['model'].encode_text(text)

    # calculate similarity
    cos_sim = CosineSimilarity(dim=1, eps=1e-6)
    dist = cos_sim(image_features, text_features)
    print(dist.item())


def main_dummy_zsl(device):
    """
    test zero-shot learning
    :return: None
    """
    quilt = load_quilt()
    data_paths = get_toy_data_paths(k=50)
    classes = ['this is invasive ductal carcinoma of the breast',
               "this is invasive lobular carcinoma of the breast",
               "this is ductal carcinoma in situ",
               'this is colorectal carcinoma',
               'this is benign breast parenchyma']
    dataset = ZeroShotDataset(quilt, data_paths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    model = ZeroShotClassifier(quilt, classes)
    model.to(device)

    for batch in dataloader:
        pred = model(batch['image'].to(device))
        print(pred['pred_class'])


if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('--device', type=str, default='cuda:0')
    PARSER.add_argument('--test_mode', type=str, default='emb', choices=['emb', 'zsl'])
    ARGS = PARSER.parse_args()
    if ARGS.test_mode == 'zsl':
        main_dummy_zsl(ARGS.device)
    elif ARGS.test_mode == 'emb':
        main_simple()
