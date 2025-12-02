# terrain-nav-preprocessing

Preprocessing tools for terrain-referenced navigation: DEM tile generation with controlled noise for enhanced computer vision feature detection.

This repository contains tools for preparing terrain data for GPS-denied UAV navigation systems using terrain-referenced navigation (TRN) and visual terrain relative navigation (VTRN) techniques.

## Features

- **DEM Download**: Automated downloading of Digital Elevation Models from OpenTopography
- **Tile Generation**: Convert DEM/imagery into compact, feature-rich tiles for onboard navigation
- **Feature Extraction**: Integrated ORB keypoint detection and VGG16-based global descriptors
- **Structure-Preserving Enhancement**: Controlled noise addition that maintains terrain correlation
- **Efficient Storage**: SQLite database with spatial indexing for fast tile retrieval

## Structure-Preserving Enhancement

<img src="method_comparison_real_dem.png" alt="Enhancement comparison" width="50%">

## Use Cases

- Terrain-referenced navigation (TRN) for GPS-denied environments
- Visual terrain relative navigation (VTRN) systems
- Place recognition and loop closure for UAV SLAM
- Onboard navigation databases with minimal storage requirements

## Requirements

### Python Dependencies

```bash
pip install requests numpy pillow opencv-python torch torchvision gdal
```

### System Requirements

- **GDAL**: Required for GeoTIFF processing
  - Ubuntu/Debian: `apt-get install gdal-bin python3-gdal`
  - macOS: `brew install gdal`
  - Windows: Use OSGeo4W installer

- **Storage**: ~10-50 MB per 100 km² depending on tile size and resolution

## Quick Start

### 1. Download DEM Data

Get a free API key from [OpenTopography](https://opentopography.org/):
1. Create an account
2. Go to "My OpenTopography" → "API Keys"
3. Generate a new key

Download DEM for your area of interest:

```bash
python download_opentopography_dem.py \
    --api-key YOUR_API_KEY \
    --west 7.916 --south 49.150 \
    --east 8.205 --north 49.319 \
    --output my_area_dem.tif
```

**Available DEM datasets:**
- `COP30`: Copernicus GLO-30 (30m resolution, recommended)
- `COP90`: Copernicus GLO-90 (90m resolution)
- `SRTMGL1`: SRTM GL1 (30m, limited coverage)
- `SRTMGL3`: SRTM GL3 (90m)
- `ALOS`: ALOS World 3D (30m)

### 2. Generate Navigation Tiles

Process the DEM into compact tiles with features:

```bash
python create_map_tiles.py \
    --dem my_area_dem.tif \
    --output ./tiles \
    --tile-size 1000
```

This creates:
- `tiles.db`: SQLite database with tile data, ORB features, and global descriptors
- `metadata.json`: Human-readable metadata about the tileset

### Optional: Add Satellite Imagery

For enhanced features, include satellite imagery:

```bash
python create_map_tiles.py \
    --dem my_area_dem.tif \
    --imagery satellite_imagery.tif \
    --output ./tiles
```

## Output Format

### SQLite Database Schema

The `tiles.db` contains:

**Tiles Table:**
- `tile_id`: Unique identifier (format: `z{zoom}_x{col}_y{row}`)
- `lat_min`, `lon_min`, `lat_max`, `lon_max`: Tile bounding box
- `dem_min_elevation`, `dem_max_elevation`: Elevation range (meters)
- `dem_data`: PNG-encoded DEM heightmap
- `imagery_data`: JPEG-encoded RGB imagery (if available)
- `orb_features`: Pickled ORB keypoints and descriptors
- `netvlad_descriptor`: 4096D float32 global descriptor
- Spatial index for fast geographic queries

### Feature Extraction Details

**ORB Features (Local):**
- 750 keypoints per tile
- Used for precise localization and terrain matching
- Structure-preserving noise (σ=12) maintains terrain correlation

**VGG16 Descriptors (Global):**
- 4096D L2-normalized vectors
- Suitable for place recognition and loop closure
- Combines max and average pooling for robustness

## Technical Details

### Structure-Preserving Noise Addition

The tile generator uses a carefully tuned noise addition method to create visual texture while preserving terrain structure:

```python
# Normalize DEM to 0-255
img_norm = ((img_array - img_array.min()) / 
            (img_array.max() - img_array.min()) * 255).astype(np.uint8)

# Add controlled noise (σ=12)
noise = (np.random.randn(*img_norm.shape) * 12).astype(np.int16)
img_array = np.clip(img_norm.astype(np.int16) + noise, 0, 255).astype(np.uint8)
```

This method achieves:
- **Correlation**: 0.199 (preserves terrain structure)
- **Feature count**: ~750 ORB keypoints per 512×512 tile
- Avoids uint8 wraparound artifacts

Compare to naive methods:
- Random noise only: r=0.014 (destroys structure)
- Histogram equalization: fewer stable features

### Tile Size Recommendations

| Tile Size | Coverage | Feature Density | Use Case |
|-----------|----------|-----------------|----------|
| 500m | Local | High | Urban navigation, high precision |
| 1000m | Regional | Medium | General UAV navigation (recommended) |
| 2000m | Wide area | Lower | Long-range flight, coarse localization |

## Example Usage in Navigation System

```python
import sqlite3
import numpy as np
import pickle

# Connect to tile database
conn = sqlite3.connect('tiles/tiles.db')
c = conn.cursor()

# Query tiles in current area
c.execute('''
    SELECT tile_id, dem_data, orb_features, netvlad_descriptor
    FROM tiles
    WHERE lat_min <= ? AND lat_max >= ? 
      AND lon_min <= ? AND lon_max >= ?
''', (lat, lat, lon, lon))

for tile_id, dem_data, orb_bytes, netvlad_bytes in c.fetchall():
    # Deserialize features
    orb_features = pickle.loads(orb_bytes)
    netvlad_desc = np.frombuffer(netvlad_bytes, dtype=np.float32)
    
    # Use for matching...
    # - ORB features for local matching
    # - NetVLAD descriptor for place recognition
```

## Performance

**Generation Speed** (on typical laptop):
- ~100 tiles/minute with DEM only
- ~50 tiles/minute with DEM + imagery
- GPU acceleration available for VGG16 feature extraction

**Storage Requirements:**
- DEM-only: ~100 KB per tile (1000m × 1000m)
- With imagery: ~200 KB per tile
- 100 km² area: ~10-20 MB total

## Command Line Reference

### download_opentopography_dem.py

```bash
python download_opentopography_dem.py [OPTIONS]

Options:
  --api-key API_KEY        OpenTopography API key (required)
  --west LONGITUDE         West boundary (default: 7.9168)
  --south LATITUDE         South boundary (default: 49.1500)
  --east LONGITUDE         East boundary (default: 8.2052)
  --north LATITUDE         North boundary (default: 49.3186)
  --output PATH            Output GeoTIFF path (default: opentopography_dem.tif)
  --dem-type TYPE          Dataset: COP30|COP90|SRTMGL1|SRTMGL3|ALOS
  --skip-verify            Skip GDAL verification

Environment Variables:
  OPENTOPOGRAPHY_API_KEY   Alternative to --api-key
```

### create_map_tiles.py

```bash
python create_map_tiles.py [OPTIONS]

Required Arguments:
  --dem PATH               Path to DEM GeoTIFF
  --output DIR             Output directory

Optional Arguments:
  --imagery PATH           Path to satellite imagery GeoTIFF
  --tile-size METERS       Tile size in meters (default: 1000)
```

## Research Background

This tool implements preprocessing techniques for terrain-based navigation in GPS-denied environments:

- **Visual-Inertial Odometry (VIO)**: Local features enable visual odometry drift correction
- **Terrain-Referenced Navigation (TRN)**: DEM matching for absolute position estimation
- **Place Recognition**: Global descriptors for loop closure and relocalization

Key papers and techniques:
- ORB features: Rublee et al., "ORB: An efficient alternative to SIFT or SURF" (ICCV 2011)
- NetVLAD: Arandjelovic et al., "NetVLAD: CNN architecture for weakly supervised place recognition" (CVPR 2016)
- Terrain navigation: Bergman (1999), "Terrain-aided navigation"

## Contributing

Contributions welcome! Areas for improvement:
- Additional DEM sources (ASTER, TanDEM-X)
- Advanced feature extractors (SuperPoint, LoFTR)
- Real-time tile streaming protocols
- Integration with SLAM frameworks

## License

MIT License - feel free to use in academic and commercial projects.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{terrain_nav_preprocessing,
  author = {Helge Stahlmann},
  title = {terrain-nav-preprocessing: Tools for GPS-Denied Navigation},
  year = {2025},
  url = {https://github.com/hstm/terrain-nav-preprocessing}
}
```

## Acknowledgments

- [OpenTopography](https://opentopography.org/) for DEM data access
- [Copernicus](https://spacedata.copernicus.eu/) for GLO-30 DEM
- Pre-trained VGG16 weights from torchvision

## Related Projects

- [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono): Visual-inertial SLAM
- [NetVLAD](https://github.com/Relja/netvlad): Original place recognition implementation
- [OpenTopography](https://opentopography.org/): Global DEM data source

---

**Questions or issues?** Open an issue on GitHub or contact [your contact info].