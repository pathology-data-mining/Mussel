from flask import Flask, request, render_template
from foundational_inference.cached_search import CachedSearch
import matplotlib.pyplot as plt
import io
import urllib, base64
from foundational_inference.utils import load_quilt
import argparse


PARSER = argparse.ArgumentParser()
PARSER.add_argument("device", help="Device to run on (cpu or cuda:{0,1,2,3})")
ARGS = PARSER.parse_args()

app = Flask(__name__)

QUILT = load_quilt()
IMG_EMB_DIR = "/gpfs/mskmind_ess/pdm/mussel_bed/quilt_1.0_224_896_None/pt_files"
PATCH_DIR = "/gpfs/mskmind_ess/pdm/mussel_bed/quilt_1.0_224_896_None/patches"
SLIDE_DIR = "/gpfs/mskmind_emc/data_large/pathology/BR_20-226/slides"
OUTPUT_DIR = "/gpfs/mskmind_ess/boehmk/scratch/queries"

CS = CachedSearch(QUILT, IMG_EMB_DIR, PATCH_DIR, SLIDE_DIR, OUTPUT_DIR, ARGS.device)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form.get('query')
        image_ids, tile_indices = CS.search(query)
        images = [CS.load_image(image_id, tile_index) for image_id, tile_index in zip(image_ids, tile_indices)]
        # Convert images to data URLs so they can be displayed in the HTML
        image_data_urls = []
        for img in images:
            if img is None:
                continue
            img_io = io.BytesIO()
            img.save(img_io, 'PNG')
            img_io.seek(0)
            image_data_urls.append('data:image/png;base64,' + urllib.parse.quote(base64.b64encode(img_io.read())))
        return render_template_string('''
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: white; color: black; display: flex; justify-content: center; align-items: center; height: 100vh; }
                form { display: flex; justify-content: center; align-items: center; }
                input[type="text"] { padding: 10px; font-size: 1.5em; }
                .image-container { display: flex; flex-wrap: wrap; justify-content: center; }
                .image-container img { margin: 10px; }
            </style>
            <form method="POST">
                <input name="query" type="text" placeholder="Enter your query here" autofocus>
            </form>
            <h1> {{ query }} </h1>
            <div class="image-container">
                {% for image_data_url in image_data_urls %}
                    <img src="{{ image_data_url }}">
                {% endfor %}
            </div>
        ''', image_data_urls=image_data_urls, query=query)
    else:
        return '''
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: white; color: black; display: flex; justify-content: center; align-items: center; height: 100vh; }
                form { display: flex; justify-content: center; align-items: center; }
                input[type="text"] { padding: 10px; font-size: 1.5em; }
            </style>
            <div class="image-container">
                <img src="{{ url_for('static', filename='msk.png') }}">
            </div>
            <div class="container">
                <h1>Welcome to Memorial Sloan Kettering's H&E image search</h1>
                <form method="POST">
                    <input name="query" type="text" placeholder="What are you looking for?" autofocus>
                </form>
            </div>
        '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
