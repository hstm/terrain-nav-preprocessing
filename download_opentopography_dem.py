#!/usr/bin/env python3
"""
Download DEM data from OpenTopography API

Downloads Copernicus GLO-30 DEM (30m resolution) for a specified bounding box.
Saves as GeoTIFF ready for processing with ORB/NetVLAD pipeline.

Requirements:
    pip install requests

Usage:
    python download_opentopography_dem.py --api-key YOUR_API_KEY
    
Get your free API key at: https://opentopography.org/
    1. Create account
    2. Go to "My OpenTopography" → "API Keys"
    3. Generate new key
"""

import requests
import os
import argparse
from pathlib import Path

def download_dem(
    west: float,
    south: float,
    east: float,
    north: float,
    output_path: str,
    api_key: str,
    dem_type: str = "COP30"
):
    """
    Download DEM from OpenTopography.
    
    Args:
        west, south, east, north: Bounding box coordinates (WGS84)
        output_path: Output GeoTIFF path
        api_key: OpenTopography API key
        dem_type: DEM dataset to use:
            - "COP30": Copernicus GLO-30 (30m, recommended)
            - "COP90": Copernicus GLO-90 (90m)
            - "SRTMGL3": SRTM GL3 (90m)
            - "SRTMGL1": SRTM GL1 (30m, limited coverage)
            - "ALOS": ALOS World 3D (30m)
    """
    
    print("OpenTopography DEM Downloader")
    print("=" * 50)
    print(f"Bounding Box:")
    print(f"  West:  {west}")
    print(f"  South: {south}")
    print(f"  East:  {east}")
    print(f"  North: {north}")
    print(f"DEM Type: {dem_type}")
    print(f"Output: {output_path}")
    print()
    
    # Calculate approximate size
    width_km = (east - west) * 111 * 0.7  # Rough approximation at ~49°N
    height_km = (north - south) * 111
    area_km2 = width_km * height_km
    print(f"Approximate area: {area_km2:.1f} km²")
    print()
    
    # OpenTopography Global DEM API endpoint
    base_url = "https://portal.opentopography.org/API/globaldem"
    
    params = {
        "demtype": dem_type,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key
    }
    
    print("Downloading DEM data...")
    print("(This may take a minute depending on area size)")
    print()
    
    try:
        response = requests.get(base_url, params=params, stream=True, timeout=300)
        response.raise_for_status()
        
        # Check if response is actually a GeoTIFF (not an error message)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type or 'application/json' in content_type:
            print(f"ERROR: API returned an error:")
            print(response.text[:500])
            return False
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Save to file
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  Downloaded: {downloaded/1024/1024:.1f} MB ({progress:.1f}%)", end="")
                    else:
                        print(f"\r  Downloaded: {downloaded/1024/1024:.1f} MB", end="")
        
        print()
        print()
        
        # Verify file
        file_size = os.path.getsize(output_path)
        if file_size < 1000:
            print(f"WARNING: File is very small ({file_size} bytes)")
            print("This might indicate an error. Check the file contents.")
            with open(output_path, 'rb') as f:
                content = f.read(500)
                if b'error' in content.lower() or b'html' in content.lower():
                    print(f"File appears to contain an error message:")
                    print(content.decode('utf-8', errors='ignore'))
                    return False
        
        print(f"✓ DEM downloaded successfully!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size / 1024 / 1024:.2f} MB")
        
        return True
        
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out. Try a smaller area or try again later.")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text[:500]}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def verify_dem(filepath: str):
    """Verify the downloaded DEM using GDAL"""
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        
        ds = gdal.Open(filepath)
        if ds is None:
            print("ERROR: Could not open file with GDAL")
            return False
        
        print()
        print("DEM Verification:")
        print(f"  Size: {ds.RasterXSize} x {ds.RasterYSize} pixels")
        print(f"  Bands: {ds.RasterCount}")
        print(f"  Projection: {ds.GetProjection()[:50]}...")
        
        gt = ds.GetGeoTransform()
        print(f"  Pixel size: {abs(gt[1]):.6f}° x {abs(gt[5]):.6f}°")
        print(f"  Resolution: ~{abs(gt[1]) * 111000:.1f}m x {abs(gt[5]) * 111000:.1f}m")
        
        band = ds.GetRasterBand(1)
        stats = band.GetStatistics(True, True)
        print(f"  Elevation range: {stats[0]:.1f}m to {stats[1]:.1f}m")
        print(f"  Mean elevation: {stats[2]:.1f}m")
        
        ds = None
        print()
        print("✓ DEM verification passed!")
        return True
        
    except ImportError:
        print("Note: GDAL not available for verification")
        return True
    except Exception as e:
        print(f"Verification error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download DEM from OpenTopography',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download with API key from command line
  python download_opentopography_dem.py --api-key YOUR_KEY
  
  # Download with API key from environment variable
  export OPENTOPOGRAPHY_API_KEY=your_key_here
  python download_opentopography_dem.py
  
  # Custom bounding box and output
  python download_opentopography_dem.py --west 8.0 --south 49.0 --east 8.5 --north 49.5 --output my_dem.tif
  
  # Use different DEM dataset
  python download_opentopography_dem.py --dem-type SRTMGL3

Get your free API key at: https://opentopography.org/
        """
    )
    
    # Bounding box (default: your specified coordinates)
    parser.add_argument('--west', type=float, default=7.916793795302511,
                       help='West longitude (default: 7.9168)')
    parser.add_argument('--south', type=float, default=49.15001303154574,
                       help='South latitude (default: 49.1500)')
    parser.add_argument('--east', type=float, default=8.20518490858376,
                       help='East longitude (default: 8.2052)')
    parser.add_argument('--north', type=float, default=49.318595052770686,
                       help='North latitude (default: 49.3186)')
    
    # Output
    parser.add_argument('--output', '-o', type=str, 
                       default='opentopography_dem.tif',
                       help='Output GeoTIFF path')
    
    # API key
    parser.add_argument('--api-key', type=str,
                       default=os.environ.get('OPENTOPOGRAPHY_API_KEY', ''),
                       help='OpenTopography API key (or set OPENTOPOGRAPHY_API_KEY env var)')
    
    # DEM type
    parser.add_argument('--dem-type', type=str, default='COP30',
                       choices=['COP30', 'COP90', 'SRTMGL1', 'SRTMGL3', 'ALOS'],
                       help='DEM dataset type (default: COP30)')
    
    # Skip verification
    parser.add_argument('--skip-verify', action='store_true',
                       help='Skip GDAL verification')
    
    args = parser.parse_args()
    
    # Check API key
    if not args.api_key:
        print("ERROR: No API key provided!")
        print()
        print("Get your free API key at: https://opentopography.org/")
        print("  1. Create an account")
        print("  2. Go to 'My OpenTopography' → 'API Keys'")
        print("  3. Generate a new key")
        print()
        print("Then run:")
        print(f"  python {os.path.basename(__file__)} --api-key YOUR_KEY")
        print()
        print("Or set environment variable:")
        print("  export OPENTOPOGRAPHY_API_KEY=your_key_here")
        return 1
    
    # Download
    success = download_dem(
        west=args.west,
        south=args.south,
        east=args.east,
        north=args.north,
        output_path=args.output,
        api_key=args.api_key,
        dem_type=args.dem_type
    )
    
    if not success:
        return 1
    
    # Verify
    if not args.skip_verify:
        verify_dem(args.output)
    
    print()
    print("=" * 50)
    print("DOWNLOAD COMPLETE!")
    print("=" * 50)
    print()
    print("Next steps:")
    print(f"  1. Use with your tile generator:")
    print(f"     python create_map_tiles.py --dem {args.output} --output ./tiles")
    print()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())