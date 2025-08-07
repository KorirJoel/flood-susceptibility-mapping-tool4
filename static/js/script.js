// Get DOM elements
const fileInput = document.getElementById('fileInput');
const csvInput = document.getElementById('csvInput'); // CSV input element
const runBtn = document.getElementById('runBtn');
const fileList = document.getElementById('fileList');
const statusMsg = document.getElementById('statusMsg');
const outputNameInput = document.getElementById('outputName');

// Array to hold selected raster files
let selectedFiles = [];

// Initialize Leaflet map centered over Kenya
const map = L.map('map').setView([0.5, 37.5], 7);

// Define basemaps
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
});
const esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  attribution: 'Tiles © Esri'
});
osm.addTo(map);  // Add OSM by default
L.control.layers({ "OpenStreetMap": osm, "Satellite (Esri)": esriSat }).addTo(map);

// For flood map overlay
let overlay;
let overlayToggle;

// When raster files are selected
fileInput.addEventListener('change', () => {
  selectedFiles = Array.from(fileInput.files).filter(f => f.name.endsWith('.tif'));
  fileList.innerHTML = '';
  selectedFiles.forEach(file => {
    const li = document.createElement('li');
    li.textContent = file.name;
    fileList.appendChild(li);
  });
});

// Run prediction when "Run" button is clicked
runBtn.addEventListener('click', async () => {
  const outputName = outputNameInput.value.trim();
  const csvFile = csvInput.files[0]; // Get the uploaded CSV file

  // Validate output name
  if (!outputName) {
    alert('Please enter an output filename.');
    return;
  }

  // Validate number of raster files
  if (selectedFiles.length !== 14) {
    alert('Please select exactly 14 .tif files.');
    return;
  }

  // Validate CSV file presence
  if (!csvFile) {
    alert('Please upload the required training CSV file.');
    return;
  }

  runBtn.disabled = true;
  statusMsg.textContent = "Training model and running prediction...";

  // Prepare FormData for sending to backend
  const formData = new FormData();
  selectedFiles.forEach(file => {
    formData.append('files[]', file);
  });
  formData.append('output_name', outputName);
  formData.append('csv_file', csvFile); // Append CSV file to the request

  try {
    // Send prediction request
    const res = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const result = await res.json();

    // Handle errors from backend
    if (!res.ok || result.error) {
      statusMsg.textContent = `Prediction failed ❌: ${result.error || 'Unknown error'}`;
      runBtn.disabled = false;
      return;
    }

    // Display prediction PNG on map
    const floodMapUrl = result.flood_map_url;
    const bounds = result.bounds || [[-4, 33], [4, 39]];

    if (overlay) {
      map.removeLayer(overlay);
      if (overlayToggle) map.removeControl(overlayToggle);
    }

    overlay = L.imageOverlay(floodMapUrl, bounds, { opacity: 0.6 });
    overlay.addTo(map);
    map.fitBounds(bounds);

    // Checkbox to toggle overlay
    overlayToggle = L.control({ position: 'topright' });
    overlayToggle.onAdd = function () {
      const div = L.DomUtil.create('div');
      div.innerHTML = `<label style="background: white; padding: 5px; border-radius: 5px;">
        <input type="checkbox" id="toggleOverlay" checked /> Show Overlay
      </label>`;
      return div;
    };
    overlayToggle.addTo(map);

    // Handle checkbox state change
    setTimeout(() => {
      const toggleCheckbox = document.getElementById('toggleOverlay');
      toggleCheckbox.addEventListener('change', function () {
        if (this.checked) {
          overlay.addTo(map);
        } else {
          map.removeLayer(overlay);
        }
      });
    }, 100);

    statusMsg.textContent = "Prediction completed ✅";

    // Save GeoTIFF blob for download
    window.floodMapTif = result.flood_map_tif;

    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.style.display = 'inline-block';
    downloadBtn.onclick = () => {
      if (window.floodMapTif) {
        // Decode base64 GeoTIFF
        const byteCharacters = atob(window.floodMapTif);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);

        // Create blob and trigger download
        const blob = new Blob([byteArray], { type: 'application/x-geotiff' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${outputName || 'flood_susceptibility_map'}.tif`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      } else {
        alert("GeoTIFF not ready for download.");
      }
    };

  } catch (error) {
    console.error("Prediction error:", error);
    statusMsg.textContent = "Prediction failed ❌";
  } finally {
    runBtn.disabled = false;
  }
});
