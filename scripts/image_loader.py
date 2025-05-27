from scripts.stackbands import stack_bands
import os

def load_image(folder_path, satellite="S2"):
        
        if satellite == "Sentinel-2" or satellite=="S2":
            required_bands = [
                "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12",
                "QA60", "MSK_CLDPRB", "SCL"  # Quality bands
            ]
        elif satellite == "Landsat 8/9" or satellite=="landsat":
              required_bands =  [
                "B01", "B02", "B03", "B04", "B05", "B06", "B07",
                "QA_PIXEL"]
        else:
            raise ValueError("Unsupported satellite type. Use 'Sentinel-2' ('S2') or 'Landsat 8/9'('landsat').")

        output_path = os.path.join(folder_path, "stacked.tif")

        stacked_path = stack_bands(folder_path, required_bands, output_path)
        
        return stacked_path

