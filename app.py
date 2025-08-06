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
matplotlib.use('Agg')  # Use non-GUI backend to avoid display issues
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import jenkspy
from jenkspy import JenksNaturalBreaks
import mapclassify
import numpy as np
           
# Initialize Flask app and enable CORS
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Ensure 'static' folder exists for storing PNG outputs
STATIC_DIR = os.path.join(os.getcwd(), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Define the expected order of 14 flood factor rasters
flood_factor_order = [
    'twi', 'tri', 'tpi', 'spi', 'soil_type', 'slope', 'profile_cu',
    'ppt', 'plan_curva', 'ndvi', 'lulc', 'elevation', 'dtoS', 'aspect'
]

# Load the pre-trained Random Forest model
model = joblib.load('rf_flood_model.pkl')

@app.route('/')
def index():
    # Renders the home page
    return render_template('index.html')

@app.route('/flood')
def flood():
    # Renders the flood prediction page
    return render_template('flood.html')

def generate_tif(profile, classified_array):
    from io import BytesIO
    import rasterio
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
    tif_base64 = base64.b64encode(tif_bytes).decode('utf-8')
    return tif_base64

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded files and output file name from form
        uploaded_files = request.files.getlist("files[]")
        output_name = request.form.get("output_name", "fsm_overlay").strip()

        if not uploaded_files:
            return jsonify({"error": "No files uploaded"}), 400

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files temporarily
            saved_paths = []
            for file in uploaded_files:
                save_path = os.path.join(temp_dir, file.filename)
                file.save(save_path)
                saved_paths.append(save_path)

            # Create a dict mapping filename (without extension) to file path
            file_dict = {
                os.path.splitext(os.path.basename(f))[0]: f
                for f in saved_paths
            }

            rasters = []
            transform = None
            dtoS_profile = None

            # Load rasters in the expected order
            for factor in flood_factor_order:
                if factor not in file_dict:
                    return jsonify({'error': f'Missing raster for: {factor}'}), 400

                with rasterio.open(file_dict[factor]) as src:
                    arr = src.read(1, resampling=Resampling.bilinear).astype(np.float32)
                    arr[arr == src.nodata] = np.nan  # Mask nodata values
                    rasters.append(arr)

                    # Save transform and profile from one layer
                    if factor == 'dtoS':
                        dtoS_profile = src.profile
                        transform = src.transform

            # Stack all layers along the last dimension
            raster_stack = np.stack(rasters, axis=-1)

            # Identify valid (non-nan) pixels
            valid_mask = ~np.any(np.isnan(raster_stack), axis=-1)
            valid_pixels = raster_stack[valid_mask]

            # Prepare data for prediction
            df_predict = pd.DataFrame(valid_pixels, columns=flood_factor_order)

            # Batch prediction for memory efficiency
            def batch_predict(X, batch_size=100000):
                return np.concatenate([
                    model.predict_proba(X[i:i + batch_size])[:, 1]
                    for i in range(0, len(X), batch_size)
                ])

            probabilities = batch_predict(df_predict)

            # Fill probability map
            prob_map = np.full(valid_mask.shape, np.nan, dtype=np.float32)
            prob_map[valid_mask] = probabilities

            # --- Reclassify using Jenks Natural Breaks --
            # Use mapclassify to compute Natural Breaks on valid data
            nb = mapclassify.NaturalBreaks(y=probabilities, k=5)  # k=number of classes

            # Use the computed bin edges to digitize the probabilities
            breaks = nb.bins  # upper bounds of each class
            classes = np.digitize(probabilities, bins=breaks)

            # Create classified output with 0 as NoData
            classified = np.full(valid_mask.shape, 0, dtype=np.uint8)

            # Assign class values 1 to 5 (Very Low to Very High)
            classified[valid_mask] = classes + 1

            # Compute geographic bounds from raster transform
            height, width = classified.shape
            west, north = transform * (0, 0)
            east, south = transform * (width, height)
            bounds = [[south, west], [north, east]]  # Format for Leaflet

            # Generate unique output PNG filename
            base_name = output_name.rstrip('.png')
            final_name = f"{base_name}.png"
            png_path = os.path.join(STATIC_DIR, final_name)
            count = 1
            while os.path.exists(png_path):
                final_name = f"{base_name}_{count}.png"
                png_path = os.path.join(STATIC_DIR, final_name)
                count += 1

            # Define class color map for visualization
            masked_data = np.ma.masked_equal(classified, 0)
            class_colors = [
                (0.0, 0.0, 0.5, 1.0),   # Dark Blue - Very Low
                (0.0, 0.5, 0.0, 1.0),   # Green - Low
                (1.0, 1.0, 0.0, 1.0),   # Yellow - Moderate
                (1.0, 0.5, 0.0, 1.0),   # Orange - High
                (1.0, 0.0, 0.0, 1.0),   # Red - Very High
            ]
            cmap = ListedColormap(class_colors)
            norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

            # Plot and save classified flood susceptibility map
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            plt.imshow(masked_data, cmap=cmap, norm=norm)

            img_buf = BytesIO()
            plt.savefig(img_buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
            img_buf.seek(0)

            # Encode the PNG image to base64 so it can be displayed in HTML
            img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
            image_data_uri = f"data:image/png;base64,{img_base64}"
            
            # Generate base64-encoded GeoTIFF
            tif_base64 = generate_tif(dtoS_profile, classified)

            # Return the base64 image URI and bounds
            return jsonify({
                'flood_map_url': image_data_uri,
                'bounds': bounds,
                'flood_map_tif': tif_base64
            })

    except Exception as e:
        # Return any errors that occur
        return jsonify({'error': str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
