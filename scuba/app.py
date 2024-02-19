from flask import Flask, request, render_template, url_for
from cached_search import CachedSearch
import matplotlib.pyplot as plt
import io
import urllib, base64
from utils import load_quilt
import argparse


PARSER = argparse.ArgumentParser()
PARSER.add_argument("device", help="Device to run on (cpu or cuda:{0,1,2,3})")
ARGS = PARSER.parse_args()

app = Flask(__name__, static_url_path='/static')

QUILT = load_quilt()
IMG_EMB_DIR = "/gpfs/mskmind_ess/pdm/reef/1.0_224_896_None/feats/quilt/pt"
PATCH_DIR = "/gpfs/mskmind_ess/pdm/reef/1.0_224_896_None/patches"
SLIDE_DIR = "/gpfs/mskmind_emc/data_large/pathology/BR_20-226/slides"
OUTPUT_DIR = "/gpfs/mskmind_ess/boehmk/scratch/queries"

CS = CachedSearch(QUILT, IMG_EMB_DIR, PATCH_DIR, SLIDE_DIR, OUTPUT_DIR, ARGS.device)

@app.route('/', methods=['GET', 'POST'])
def home():
    image_data_urls = []
    query = ""
    if request.method == 'POST':
        query = request.form.get('query')
        image_ids, tile_indices = CS.search(query)
        images = [CS.load_image(image_id, tile_index) for image_id, tile_index in zip(image_ids, tile_indices)]
        # Convert images to data URLs so they can be displayed in the HTML
        for img in images:
            if img is None:
                continue
            img_io = io.BytesIO()
            img.save(img_io, 'PNG')
            img_io.seek(0)
            image_data_urls.append('data:image/png;base64,' + urllib.parse.quote(base64.b64encode(img_io.read())))
    return render_template('index.html', image_data_urls=image_data_urls, num_slides=CS.len, query=query)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
