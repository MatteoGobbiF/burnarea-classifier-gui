import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import rasterio as rio
from scripts.preprocessing import detect_nodata, select_spectral_bands
import os
import joblib


def compute_dNBR(pre_NBR, post_NBR):
    # Ensure inputs are NumPy arrays
    pre_NBR = np.asarray(pre_NBR, dtype=np.float32)
    post_NBR = np.asarray(post_NBR, dtype=np.float32)
    
    # Check for shape mismatch
    if pre_NBR.shape != post_NBR.shape:
        raise ValueError(f"Shape mismatch: pre_NBR {pre_NBR.shape} and post_NBR {post_NBR.shape} must be the same.")

    # Compute dNBR
    dNBR = (pre_NBR - post_NBR) * 1000

    # Handle NaN propagation explicitly (NumPy does this by default, but being explicit is good)
    dNBR[np.isnan(pre_NBR) | np.isnan(post_NBR)] = np.nan

    # Optional: Handle extreme values (if necessary)
    dNBR[np.isinf(dNBR)] = np.nan  # Convert infinities to NaN

    return dNBR

def create_training_set(post_image_path, dNBR, method='threshold', threshold=300, ext_burned_threshold=500, ext_unburned_threshold=50, samples=5000, post_image_nodata=None):
    # Load the post-fire image
    with rio.open(post_image_path) as src:
        band_descriptions = src.descriptions  # Get all band names
        selected_bands = select_spectral_bands(band_descriptions)  # Filter spectral bands
        band_indices = [band_descriptions.index(b) + 1 for b in selected_bands]  # Convert to 1-based index

        # Read only the selected spectral bands
        post_image = src.read(band_indices).astype(np.float32)
        if post_image_nodata is None:
            post_image_nodata = detect_nodata(post_image, src.nodata)

    if method == 'extreme':
        burned_area = compute_burned_area_extreme(dNBR, ext_burned_threshold, ext_unburned_threshold)
    elif method == 'threshold':
        burned_area = compute_burned_area_threshold(dNBR, threshold)
    else: 
        raise ValueError("Invalid method! Choose 'threshold' or 'extreme'.")
    
    # Reshape features from (bands, height, width) to (pixels, bands)
    features_reshaped = post_image.reshape(post_image.shape[0], -1).T  # Transpose to (pixels, bands)

    # Flatten labels from (height, width) to (pixels,)
    labels_reshaped = burned_area.flatten()

    if post_image_nodata is not None:
        post_image_nodata=float(post_image_nodata)

    if post_image_nodata is not None and np.isnan(post_image_nodata):
        valid_mask = ~np.isnan(features_reshaped).any(axis=1)
    elif post_image_nodata is not None:
        valid_mask = ~(features_reshaped == post_image_nodata).any(axis=1)
    else:
        valid_mask = np.ones(features_reshaped.shape[0], dtype=bool)  # Keep all if nodata is None

    # Ensure labels (y) are not NaN
    valid_label_mask = ~np.isnan(labels_reshaped)

    # Combine both masks
    valid_mask = valid_mask & valid_label_mask  

    # Filter features and labels using valid pixels
    X = features_reshaped[valid_mask]
    y = labels_reshaped[valid_mask]

    print(f"Extracted {X.shape[0]} valid samples with {X.shape[1]} features each.")

    # Ensure we don't sample more than available data
    samples = min(samples, X.shape[0])

    # Randomly sample training points
    X_sampled, y_sampled = resample(X, y, n_samples=samples, random_state=42)

    print(f"Sampled {X_sampled.shape[0]} points for training.")

    return X_sampled, y_sampled

def train_and_evaluate_svm(X_sampled, y_sampled, test_size=0.25, kernel="linear", random_state=42):

    scaler = MinMaxScaler(feature_range=(0,1))
    X_sampled = scaler.fit_transform(X_sampled)

    # Split sampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=test_size, random_state=random_state)

    # Train the SVM
    svm = SVC(kernel=kernel, random_state=random_state)
    svm.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm.predict(X_test)

    # Compute evaluation metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)  # Convert to dict
    accuracy = accuracy_score(y_test, y_pred)
    '''
    print("Confusion Matrix:")
    print(conf_matrix)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nAccuracy: {accuracy:.4f}")
    '''
    return svm, conf_matrix, class_report, accuracy  # Return all metrics

def test_model(model, X_test, y_test):

    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)

    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)  # Convert to dict
    accuracy = accuracy_score(y_test, y_pred)

    return conf_matrix, class_report, accuracy  # Return all metrics

def save_model(model, model_path):

    joblib.dump(model, model_path)
    print(f"SVM model saved to {model_path}")

def load_model(model_path):

    model = joblib.load(model_path)
    print(f"SVM model loaded from {model_path}")
    return model
    

def compute_burned_area_threshold(dNBR, threshold=300):
    burned_area = np.where(np.isnan(dNBR), np.nan, np.where(dNBR >= threshold, 1, 0))
    return burned_area

def compute_burned_area_extreme(dNBR, b_threshold=500, ub_threshold=50):
    burned_area = np.full(dNBR.shape, np.nan)  # Initialize as NaN
    burned_area[dNBR > b_threshold] = 1  # Burned
    burned_area[dNBR < ub_threshold] = 0  # Unburned
    return burned_area

def classify(model, post_image_path, output_path=None, nodata_value=None):

    with rio.open(post_image_path) as src:
        band_descriptions = src.descriptions  # Get all band names
        selected_bands = select_spectral_bands(band_descriptions)  # Filter spectral bands
        band_indices = [band_descriptions.index(b) + 1 for b in selected_bands]  # Convert to 1-based index

        # Read only the selected spectral bands
        post_image = src.read(band_indices).astype(np.float32)
        features_meta = src.profile  # Get metadata

        # Try to read NoData from metadata
        image_nodata = src.nodata

    # Detect NoData value (use user input, metadata, or guess)
    final_nodata = nodata_value if nodata_value is not None else detect_nodata(post_image, image_nodata)

    # Generate default output filename if not provided
    if output_path is None:
            file_dir, file_name = os.path.split(post_image_path)
            file_base, _ = os.path.splitext(file_name)
            output_path = os.path.join(file_dir, f"{file_base}_classified.tif")

    # Reshape features for classification: (bands, height, width) → (pixels, bands)
    features_reshaped = post_image.reshape(post_image.shape[0], -1).T  # (pixels, bands)

    if final_nodata is not None and np.isnan(final_nodata):
        valid_mask = ~np.isnan(features_reshaped).any(axis=1)
    elif final_nodata is not None:
        valid_mask = ~(features_reshaped == final_nodata).any(axis=1)
    else:
        valid_mask = np.ones(features_reshaped.shape[0], dtype=bool)
        

    # Standardize feature values
    scaler = MinMaxScaler()
    features_reshaped = scaler.fit_transform(features_reshaped)
    

    # Initialize an empty array for classification results
    classified_raster = np.full(features_reshaped.shape[0], final_nodata)  # Preserve NoData

    # Classify only valid pixels
    classified_raster[valid_mask] = model.predict(features_reshaped[valid_mask])

    # Reshape back to raster dimensions (height, width)
    classified_raster = classified_raster.reshape(post_image.shape[1], post_image.shape[2])

    # Update metadata for output raster
    features_meta.update({
        "driver": "GTiff",
        "count": 1,  # Single-band classification output
        "dtype": "float32",  # Keep as float to maintain NaNs
        "nodata": final_nodata  # Preserve detected NoData value
    })

    # Save the classified raster
    with rio.open(output_path, "w", **features_meta) as dst:
        dst.write(classified_raster.astype(np.float32), 1)

    print(f"Classified raster saved to {output_path}")
    
    return output_path
