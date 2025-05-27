import geopre as gp
import numpy as np
import rasterio as rio
import os
import shutil

'''
def clip_image(source, geometry):
    with fiona.open(geometry, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    
    with rio.open("tests/data/RGB.byte.tif") as src:
        out_image, out_transform = rio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    
    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    with rio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
        dest.write(out_image)

'''

def mask_clouds_any(source, output_path, satellite, method, mask_shadows, nodata=np.nan):
    if satellite == 'S2':
        return gp.mask_clouds_S2(source, output_path, method=method, mask_shadows=mask_shadows, nodata_value=nodata)
    elif satellite =='L8':
        return gp.mask_clouds_landsat(source, output_path, method=method, mask_shadows=mask_shadows, nodata_value=nodata)
    return output_path

def mask_water_any(source, output_path, satellite, ndwi_threshold, nodata=np.nan):
    if output_path is None:
        file_dir, file_name = os.path.split(source)
        file_base, _ = os.path.splitext(file_name)  # Ignore original extension
        output_path = os.path.join(file_dir, f"{file_base}_temp_water_masked.tif")

    if satellite == 'S2':
        return gp.mask_water_S2(source, output_path, ndwi_threshold, nodata)
    elif satellite == 'L8':
        return gp.mask_water_landsat(source, output_path, ndwi_threshold, nodata)
    return output_path

def mask_water_S2(image_path, output_path = None, ndwi_threshold=0.1, nodata = np.nan):
    # Open the input image

    if output_path is None:
        file_dir, file_name = os.path.split(image_path)
        file_base, _ = os.path.splitext(file_name)  # Ignore original extension
        output_path = os.path.join(file_dir, f"{file_base}_temp_water_masked.tif")

    with rio.open(image_path) as src:
        band_descriptions = src.descriptions  # Get band descriptions
        image = src.read()  # Read all bands into a NumPy array
        meta = src.meta.copy()  # Copy metadata

        # Get indices for B3 (Green) and B8/B8A (NIR)
        try:
            if "B3" in band_descriptions:
                b3_idx = band_descriptions.index("B3") + 1 
            elif "B03" in band_descriptions:
                b3_idx = band_descriptions.index("B03") + 1
            else:
                raise ValueError("Band 3 not found in band descriptions!")
            if "B8" in band_descriptions:
                b8_idx = band_descriptions.index("B8") + 1
            elif "B08" in band_descriptions:
                b8_idx = band_descriptions.index("B08") + 1
            elif "B8A" in band_descriptions:
                b8_idx = band_descriptions.index("B8A") + 1
            else:
                raise ValueError("Neither B8 nor B8A found in band descriptions!")
        except ValueError as e:
            raise ValueError(f"Required band(s) not found: {e}")

        # Read Green (B3) and NIR (B8 or B8A) bands
        green = src.read(b3_idx).astype(np.float32)
        nir = src.read(b8_idx).astype(np.float32)

        # Compute NDWI = (Green - NIR) / (Green + NIR)
        ndwi = (green - nir) / (green + nir + 1e-10)  # Avoid division by zero

        # Create water mask (non water: NDWI < 0.01)
        water_mask = ndwi < 0.01

        # Apply the mask to all bands
        masked_image = np.where(water_mask, image, nodata)

        # Update metadata
        meta.update({"nodata": nodata, "dtype": 'float32'})

    # Save the masked image
    with rio.open(output_path, "w", **meta) as dest:
        dest.write(masked_image.astype('float32'))
        
        # Preserve band descriptions in the output file
        dest.descriptions = band_descriptions

    return output_path  # Return the new file path for reference

def mask_water_landsat(image_path, output_path = None, ndwi_threshold=0.1, nodata = np.nan):

    if output_path is None:
        file_dir, file_name = os.path.split(image_path)
        file_base, _ = os.path.splitext(file_name)  # Ignore original extension
        output_path = os.path.join(file_dir, f"{file_base}_temp_water_masked.tif")

    # Open the input image
    with rio.open(image_path) as src:
        band_descriptions = src.descriptions  # Get band descriptions
        image = src.read()  # Read all bands into a NumPy array
        meta = src.meta.copy()  # Copy metadata

        # Get indices for Green (B3) and NIR (B5) bands in Landsat 8
        try:
            if "B3" in band_descriptions:
                b3_idx = band_descriptions.index("B3") + 1  # Rasterio is 1-based
            elif "B03" in band_descriptions:
                b3_idx = band_descriptions.index("B03") + 1
            elif "SR_B3" in band_descriptions:
                b3_idx = band_descriptions.index("SR_B3") + 1
            else:
                raise ValueError("Band B3 not found in band descriptions!")

            if "B5" in band_descriptions:
                b5_idx = band_descriptions.index("B5") + 1
            elif "B05" in band_descriptions:
                b5_idx = band_descriptions.index("B05") + 1
            elif "SR_B5" in band_descriptions:
                b5_idx = band_descriptions.index("SR_B5") + 1
            else:
                raise ValueError("Band B5 not found in band descriptions!")
        except ValueError as e:
            raise ValueError(f"Required band(s) not found: {e}")

        # Read Green (B3) and NIR (B5) bands
        green = src.read(b3_idx).astype(np.float32)
        nir = src.read(b5_idx).astype(np.float32)

        # Compute NDWI = (Green - NIR) / (Green + NIR)
        ndwi = (green - nir) / (green + nir + 1e-10)  # Avoid division by zero

        # Create water mask (water: NDWI < 0.01)
        water_mask = ndwi < ndwi_threshold

        # Apply the mask to all bands
        masked_image = np.where(water_mask, image, nodata)

        # Update metadata
        meta.update({"nodata": nodata, "dtype": 'float32'})

    # Save the masked image
    with rio.open(output_path, "w", **meta) as dest:
        dest.write(masked_image)
        
        # Preserve band descriptions in the output file
        dest.descriptions = band_descriptions

    return output_path  # Return the new file path for reference

def apply_masks(image_path, satellite, final_output_path = None, method='auto', mask_clouds=True, mask_shadows=True, mask_water=True, ndwi_threshold=0.01, nodata=None):
    """Apply water masking first, then call the cloud masking function."""
    
    # Define the final output file (always save as .tif)
    if final_output_path is None:
        file_dir, file_name = os.path.split(image_path)
        file_base, _ = os.path.splitext(file_name)  # Ignore original extension
        final_output_path = os.path.join(file_dir, f"{file_base}_masked.tif")

    final_nodata = nodata if nodata is not None else detect_nodata_from_path(image_path)
    if final_nodata is None:
        final_nodata = np.nan  # Default to NaN if no NoData was detected


    if mask_water:
        '''
        if satellite=='S2':
            temp_water_masked = mask_water_S2(image_path, ndwi_threshold=ndwi_threshold, nodata=final_nodata)  # Save a temp water-masked image
        elif satellite=='L8':
            temp_water_masked = mask_water_landsat(image_path, ndwi_threshold=ndwi_threshold, nodata=final_nodata)
        '''
        temp_water_masked = mask_water_any(image_path, output_path=None, satellite=satellite, ndwi_threshold=ndwi_threshold, nodata=final_nodata)
        if not mask_clouds:
            shutil.move(temp_water_masked, final_output_path)  # Rename/move file
    else:
        temp_water_masked = image_path 

    if mask_clouds:
        cloud_masked_path = mask_clouds_any(temp_water_masked, final_output_path, satellite, method, mask_shadows, final_nodata)
        final_output_path = cloud_masked_path
        # Clean up the temp water-masked file (if it was different from the original)
        if mask_water and temp_water_masked != image_path and os.path.exists(temp_water_masked):
            os.remove(temp_water_masked)

    return final_output_path  # Return the final processed image

def compute_NBR(source, satellite, nodata=None):

    try:
        with rio.open(source) as src:
            band_descriptions = src.descriptions  # Get band descriptions
            # Define band indices based on satellite type
            if satellite == 'S2':  # Sentinel-2
                nir_band = ["B8A", "B08", "B8"]  # Check for different NIR names
                swir2_band = ["B12"]
            elif satellite == 'L8':  # Landsat 8/9
                nir_band = ["B5", "B05", "SR_B5"]
                swir2_band = ["B7", "B07", "SR_B7"]
            else:
                raise ValueError("Invalid satellite type! Choose 'S2' for Sentinel-2 or 'L8' for Landsat 8/9.")

            # Get correct band indices
            nir_idx = next((band_descriptions.index(b) + 1 for b in nir_band if b in band_descriptions), None)
            swir2_idx = next((band_descriptions.index(b) + 1 for b in swir2_band if b in band_descriptions), None)

            if nir_idx is None or swir2_idx is None:
                raise ValueError(f"Required bands not found in {satellite} image: {nir_band} & {swir2_band}")

            # Read the bands
            nir = src.read(nir_idx).astype(np.float32)
            swir2 = src.read(swir2_idx).astype(np.float32)

            if nodata is None:
                # Try detecting nodata from image
                nodata = detect_nodata(src.read(), src.nodata)  # Check using NIR band
            nodata = float(nodata) if nodata is not None else None  # Convert to float if detected
            
            # Apply nodata masking if detected
            if nodata is not None:
                nir[nir == nodata] = np.nan
                swir2[swir2 == nodata] = np.nan

            # Compute NBR
            nbr = (nir - swir2) / (nir + swir2 + 1e-10)  # Avoid division by zero

            return nbr  # Return computed NBR array
        
    except Exception as e:
        print(f"Error computing NBR for {satellite}: {e}")
        return None  # Return None if an error occurs


def detect_nodata_from_path(image_path):
    with rio.open(image_path) as src:
        return detect_nodata(src.read(), src.nodata)

def detect_nodata(image, metadata_nodata=None):

    if metadata_nodata is not None:
        return metadata_nodata  # Use NoData from metadata if provided

    # Convert to float to safely check for NaNs
    image = image.astype(np.float32)

    # Check if at least one pixel has -9999 across all bands
    if np.any(np.all(image == -9999, axis=0)):
        return -9999  

    # Check if at least one pixel has 0 across all bands
    if np.any(np.all(image == 0, axis=0)):
        return 0  

    # Check if at least one pixel has NaN across all bands
    if np.any(np.all(np.isnan(image), axis=0)):
        return np.nan 
    
    return None # If nothing is detected

def select_spectral_bands(band_descriptions):
    """Filters out non-spectral bands, keeping only valid reflectance bands."""
    spectral_bands = []

    for band in band_descriptions:
        # Allow:
        # - Sentinel-2: "B1", "B8A", "B12"
        # - Landsat 8 SR: "SR_B1", "SR_B2", ..., "SR_B7", "SR_B10", "SR_B11"
        if (band.startswith("B") or band.startswith("SR_B")) and any(char.isdigit() for char in band):
            spectral_bands.append(band)

    return spectral_bands