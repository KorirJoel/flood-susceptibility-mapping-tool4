from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import joblib
from sklearn.ensemble import RandomForestClassifier
import tempfile
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import traceback
import mapclassify

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Ensure 'static' folder exists
STATIC_DIR = os.path.join(os.getcwd(), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Expected raster order (names must match model training columns exactly!)
flood_factor_order = [
    'twi', 'tri', 'tpi', 'spi', 'soil_type', 'slope', 'profile_cu',
    'ppt', 'plan_curva', 'ndvi', 'lulc', 'elevation', 'dtoS', 'aspect'
]

# Load model (ensure correct path)
model_path = os.path.join(os.path.dirname(__file__), 'rf_flood_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/flood')
def flood():
    return render_template('flood.html')

def generate_tif(profile, classified_array):
    from rasterio.io import MemoryFile

    profile.update({
        'dtype': rasterio.uint8,
        'count': 1,
        'compress': 'lzw',
        'nodata': 0
    })

    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(classified_array, 1)

    tif_bytes = memfile.read()
    return base64.b64encode(tif_bytes).decode('utf-8')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_files = request.files.getlist("files[]")
        output_name = request.form.get("output_name", "fsm_overlay").strip()

        if not uploaded_files:
            return jsonify({"error": "No files uploaded"}), 400

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = []
            for file in uploaded_files:
                save_path = os.path.join(temp_dir, file.filename)
                file.save(save_path)
                saved_paths.append(save_path)

            # Match filenames case-insensitively, but preserve model feature names
            file_dict_raw = {
                os.path.splitext(os.path.basename(f))[0].lower(): f
                for f in saved_paths
            }

            file_dict = {}
            for expected_name in flood_factor_order:
                match = file_dict_raw.get(expected_name.lower())
                if not match:
                    return jsonify({'error': f'Missing raster for: {expected_name}'}), 400
                file_dict[expected_name] = match

            print("Matched raster files to features:", list(file_dict.keys()))

            rasters = []
            transform = None
            dtoS_profile = None

            for factor in flood_factor_order:
                with rasterio.open(file_dict[factor]) as src:
                    arr = src.read(1, resampling=Resampling.bilinear).astype(np.float32)
                    arr[arr == src.nodata] = np.nan
                    rasters.append(arr)

                    print(f"Loaded '{factor}' raster - shape: {arr.shape}, NaNs: {np.isnan(arr).sum()}")

                    if factor == 'dtoS':
                        dtoS_profile = src.profile
                        transform = src.transform

            raster_stack = np.stack(rasters, axis=-1)
            valid_mask = ~np.any(np.isnan(raster_stack), axis=-1)
            valid_pixels = raster_stack[valid_mask]

            if valid_pixels.size == 0:
                raise ValueError("No valid pixels found for prediction.")

            df_predict = pd.DataFrame(valid_pixels, columns=flood_factor_order)

            def batch_predict(X, batch_size=100000):
                return np.concatenate([
                    model.predict_proba(X[i:i + batch_size])[:, 1]
                    for i in range(0, len(X), batch_size)
                ])

            probabilities = batch_predict(df_predict)
            print("Prediction complete. Min prob:", np.min(probabilities), "Max prob:", np.max(probabilities))

            prob_map = np.full(valid_mask.shape, np.nan, dtype=np.float32)
            prob_map[valid_mask] = probabilities

            nb = mapclassify.NaturalBreaks(y=probabilities, k=5)
            breaks = nb.bins
            classes = np.digitize(probabilities, bins=breaks)
            classified = np.full(valid_mask.shape, 0, dtype=np.uint8)
            classified[valid_mask] = classes + 1

            height, width = classified.shape
            west, north = transform * (0, 0)
            east, south = transform * (width, height)
            bounds = [[south, west], [north, east]]

            base_name = output_name.rstrip('.png')
            final_name = f"{base_name}.png"
            png_path = os.path.join(STATIC_DIR, final_name)
            count = 1
            while os.path.exists(png_path):
                final_name = f"{base_name}_{count}.png"
                png_path = os.path.join(STATIC_DIR, final_name)
                count += 1

            masked_data = np.ma.masked_equal(classified, 0)
            class_colors = [
                (0.0, 0.0, 0.5, 1.0),   # Dark Blue
                (0.0, 0.5, 0.0, 1.0),   # Green
                (1.0, 1.0, 0.0, 1.0),   # Yellow
                (1.0, 0.5, 0.0, 1.0),   # Orange
                (1.0, 0.0, 0.0, 1.0),   # Red
            ]
            cmap = ListedColormap(class_colors)
            norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

            plt.figure(figsize=(10, 8))
            plt.axis('off')
            plt.imshow(masked_data, cmap=cmap, norm=norm)

            img_buf = BytesIO()
            plt.savefig(img_buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
            img_buf.seek(0)

            img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
            image_data_uri = f"data:image/png;base64,{img_base64}"
            tif_base64 = generate_tif(dtoS_profile, classified)

            return jsonify({
                'flood_map_url': image_data_uri,
                'bounds': bounds,
                'flood_map_tif': tif_base64
            })

    except Exception as e:
        print("Prediction failed:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
