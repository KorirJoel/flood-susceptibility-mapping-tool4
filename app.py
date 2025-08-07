# Import required libraries
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestClassifier
import tempfile
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import mapclassify  # For Jenks Natural Breaks classification

# Initialize Flask app and enable CORS for cross-origin requests
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Ensure static folder exists for storing generated images
STATIC_DIR = os.path.join(os.getcwd(), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Ordered list of expected raster layers (by name without extension)
flood_factor_order = [
    'twi', 'tri', 'tpi', 'spi', 'soil_type', 'slope', 'profile_cu',
    'ppt', 'plan_curva', 'ndvi', 'lulc', 'elevation', 'dtoS', 'aspect'
]

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for flood prediction UI page
@app.route('/flood')
def flood():
    return render_template('flood.html')

# Utility function to generate a base64-encoded GeoTIFF from array and profile
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

# Route for flood susceptibility prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded raster files and CSV training file from request
        uploaded_files = request.files.getlist("files[]")
        csv_file = request.files.get("csv_file")
        output_name = request.form.get("output_name", "fsm_overlay").strip()

        # Ensure required files are provided
        if not uploaded_files or not csv_file:
            return jsonify({"error": "Missing raster files or CSV file"}), 400

        # Load training data from CSV
        train_df = pd.read_csv(csv_file)
        X_train = train_df[flood_factor_order]
        y_train = train_df['status']

        # Train the Random Forest model on CSV data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded raster files to temporary directory
            saved_paths = []
            for file in uploaded_files:
                save_path = os.path.join(temp_dir, file.filename)
                file.save(save_path)
                saved_paths.append(save_path)

            # Create dictionary mapping raster names to file paths
            file_dict = {
                os.path.splitext(os.path.basename(f))[0]: f for f in saved_paths
            }

            rasters = []
            transform = None
            dtoS_profile = None

            # Read raster layers in the expected order
            for factor in flood_factor_order:
                if factor not in file_dict:
                    return jsonify({'error': f'Missing raster for: {factor}'}), 400

                with rasterio.open(file_dict[factor]) as src:
                    arr = src.read(1, resampling=Resampling.bilinear).astype(np.float32)
                    arr[arr == src.nodata] = np.nan  # Handle nodata values
                    rasters.append(arr)
                    if factor == 'dtoS':  # Save one profile for output
                        dtoS_profile = src.profile
                        transform = src.transform

            # Stack rasters into a 3D array (rows, cols, bands)
            raster_stack = np.stack(rasters, axis=-1)

            # Mask to identify valid (non-NaN) pixels
            valid_mask = ~np.any(np.isnan(raster_stack), axis=-1)

            # Extract valid pixels for prediction
            valid_pixels = raster_stack[valid_mask]
            df_predict = pd.DataFrame(valid_pixels, columns=flood_factor_order)

            # Predict probabilities in batches for memory efficiency
            def batch_predict(X, batch_size=100000):
                return np.concatenate([
                    model.predict_proba(X[i:i + batch_size])[:, 1]
                    for i in range(0, len(X), batch_size)
                ])

            probabilities = batch_predict(df_predict)

            # Initialize full-size probability map and assign values
            prob_map = np.full(valid_mask.shape, np.nan, dtype=np.float32)
            prob_map[valid_mask] = probabilities

            # Classify predicted probabilities using Jenks Natural Breaks
            nb = mapclassify.NaturalBreaks(y=probabilities, k=5)
            breaks = nb.bins
            classes = np.digitize(probabilities, bins=breaks)

            # Assign classified values (1–5) to valid pixels
            classified = np.full(valid_mask.shape, 0, dtype=np.uint8)
            classified[valid_mask] = classes + 1  # Add 1 so classes range 1–5

            # Get map bounds in lat/lon from transform
            height, width = classified.shape
            west, north = transform * (0, 0)
            east, south = transform * (width, height)
            bounds = [[south, west], [north, east]]

            # Determine unique PNG file name to avoid overwrite
            base_name = output_name.rstrip('.png')
            final_name = f"{base_name}.png"
            png_path = os.path.join(STATIC_DIR, final_name)
            count = 1
            while os.path.exists(png_path):
                final_name = f"{base_name}_{count}.png"
                png_path = os.path.join(STATIC_DIR, final_name)
                count += 1

            # Mask and visualize the classified map
            masked_data = np.ma.masked_equal(classified, 0)
            class_colors = [
                (0.0, 0.0, 0.5, 1.0),  # Very Low
                (0.0, 0.5, 0.0, 1.0),  # Low
                (1.0, 1.0, 0.0, 1.0),  # Moderate
                (1.0, 0.5, 0.0, 1.0),  # High
                (1.0, 0.0, 0.0, 1.0),  # Very High
            ]
            cmap = ListedColormap(class_colors)
            norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

            # Plot classified map
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            plt.imshow(masked_data, cmap=cmap, norm=norm)

            # Save plot to PNG and encode as base64
            img_buf = BytesIO()
            plt.savefig(img_buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
            image_data_uri = f"data:image/png;base64,{img_base64}"

            # Generate encoded GeoTIFF
            tif_base64 = generate_tif(dtoS_profile, classified)

            # Return base64 image + bounds + tif data to frontend
            return jsonify({
                'flood_map_url': image_data_uri,
                'bounds': bounds,
                'flood_map_tif': tif_base64
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app locally
if __name__ == '__main__':
    app.run(debug=True)
