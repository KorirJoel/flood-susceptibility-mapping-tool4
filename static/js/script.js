const fileInput = document.getElementById('fileInput');
const runBtn = document.getElementById('runBtn');
const fileList = document.getElementById('fileList');
const statusMsg = document.getElementById('statusMsg');
const outputNameInput = document.getElementById('outputName');

let selectedFiles = [];

// Initialize Leaflet map centered over Kenya
const map = L.map('map').setView([0.5, 37.5], 7);

const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
});
const esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  attribution: 'Tiles © Esri'
});

osm.addTo(map);
L.control.layers({ "OpenStreetMap": osm, "Satellite (Esri)": esriSat }).addTo(map);

let overlay;
let overlayToggle;

fileInput.addEventListener('change', () => {
  selectedFiles = Array.from(fileInput.files).filter(f => f.name.endsWith('.tif'));
  fileList.innerHTML = '';
  selectedFiles.forEach(file => {
    const li = document.createElement('li');
    li.textContent = file.name;
    fileList.appendChild(li);
  });
});

runBtn.addEventListener('click', async () => {
  const outputName = outputNameInput.value.trim();

  if (!outputName) {
    alert('Please enter an output filename.');
    return;
  }

  if (selectedFiles.length !== 14) {
    alert('Please select exactly 14 .tif files.');
    return;
  }

  runBtn.disabled = true;
  statusMsg.textContent = "Running prediction...";

  const formData = new FormData();
  selectedFiles.forEach(file => {
    formData.append('files[]', file);
  });
  formData.append('output_name', outputName);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const result = await res.json();

    if (!res.ok || result.error) {
      statusMsg.textContent = `Prediction failed ❌: ${result.error || 'Unknown error'}`;
      runBtn.disabled = false;
      return;
    }

    const floodMapUrl = result.flood_map_url; // PNG for visualization
    const bounds = result.bounds || [[-4, 33], [4, 39]];

    // Map overlay preview (optional)
    if (overlay) {
      map.removeLayer(overlay);
      if (overlayToggle) map.removeControl(overlayToggle);
    }

    overlay = L.imageOverlay(floodMapUrl, bounds, { opacity: 0.6 });
    overlay.addTo(map);
    map.fitBounds(bounds);

    overlayToggle = L.control({ position: 'topright' });
    overlayToggle.onAdd = function () {
      const div = L.DomUtil.create('div');
      div.innerHTML = `<label style="background: white; padding: 5px; border-radius: 5px;">
        <input type="checkbox" id="toggleOverlay" checked /> Show Overlay
      </label>`;
      return div;
    };
    overlayToggle.addTo(map);

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

    // ✅ Handle GeoTIFF download only
    window.floodMapTif = result.flood_map_tif;

    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.style.display = 'inline-block';
    downloadBtn.onclick = () => {
      if (window.floodMapTif) {
        // Decode the base64 string
        const byteCharacters = atob(window.floodMapTif);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);

        // Create Blob and URL
        const blob = new Blob([byteArray], { type: 'application/x-geotiff' });
        const url = URL.createObjectURL(blob);

        // Create anchor tag and trigger download
        const link = document.createElement('a');
        link.href = url;
        link.download = `${outputName || 'flood_susceptibility_map'}.tif`;
        document.body.appendChild(link);
        link.click();

        // Clean up
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
