import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Toplevel
from scripts import preprocessing
from scripts import processing
from scripts import image_loader
import rasterio as rio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import threading
import numpy as np

class BurnAreaClassifierApp:
    def __init__(self, root):
        """Initialize the main window"""
        self.root = root
        self.root.title("Burned Area Classification Tool")  
        self.root.geometry("400x300")  # Adjusted window size

        ttk.Label(root, text="Select an Option:", font=("Arial", 14)).pack(pady=10)

        # Buttons for mode selection
        ttk.Button(root, text="Classify (Train New Model)", command=self.train_new_model, width=30).pack(pady=5)
        ttk.Button(root, text="Classify (Use Existing Model)", command=self.use_existing_model, width=30).pack(pady=5)
        ttk.Button(root, text="Test Performance of Model", command=self.test_model_performance, width=30).pack(pady=5)

    def train_new_model(self):
        """Go to Train New Model mode"""
        self.clear_window()
        TrainModelUI(self.root)

    def use_existing_model(self):
        """Go to Use Existing Model mode"""
        self.clear_window()
        UseExistingModelUI(self.root)

    def test_model_performance(self):
        """Go to Test Model Performance mode"""
        self.clear_window()
        TestModelPerformanceUI(self.root)

    def clear_window(self):
        """Destroy all widgets before switching mode"""
        for widget in self.root.winfo_children():
            widget.destroy()

class TrainModelUI:
    def __init__(self, root):
        """UI for model training"""
        self.root = root
        self.root.title("Train New Model")
        self.root.geometry("650x850")
        root.resizable(False, False)
    
        
        label_width = 22
        
        # Satellite Selection
        ttk.Label(root, text="Select Satellite:", width=label_width, anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.satellite_var = tk.StringVar(value="Sentinel-2")
        self.satellite_dropdown = ttk.Combobox(root, textvariable=self.satellite_var, values=["Sentinel-2", "Landsat 8/9"], state="readonly", width=15)
        self.satellite_dropdown.grid(row=0, column=0, padx=100, pady=5, sticky="w")
        self.satellite_dropdown.bind("<<ComboboxSelected>>", self.update_cloud_methods)
        
        # Frame for Before Image Processing
        self.pre_frame = ttk.LabelFrame(root, text="Before Image", padding=(10, 5, 10, 5))
        self.pre_frame.grid(row=1, column=0, columnspan=5, padx=10, pady=10, sticky="ew")

        # Before Image Selection
        ttk.Label(self.pre_frame, text="Select Before Image:", width=label_width, anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pre_entry = ttk.Entry(self.pre_frame, width=50)
        self.pre_entry.grid(row=0, column=1, padx=5, pady=5)
        self.pre_button = ttk.Button(self.pre_frame, text="...", command=self.load_pre_image, width=3)
        self.pre_button.grid(row=0, column=2, padx=5, pady=5)
        self.pre_button.grid(row=0, column=2, padx=5, pady=5)
        self.pre_folder_button = ttk.Button(self.pre_frame, text="📁", command=self.load_pre_folder, width=3)
        self.pre_folder_button.grid(row=0, column=3, padx=5, pady=5)

        # Pre Image NoData
        ttk.Label(self.pre_frame, text="Nodata value:", width=label_width, anchor="w").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.pre_nodata_entry = ttk.Entry(self.pre_frame, width=7)
        self.pre_nodata_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Before Image Preview
        self.pre_preview_button = ttk.Button(self.pre_frame, text="👁️", command=self.pre_preview, width=3)
        self.pre_preview_button.grid(row=0, column=4, padx=5, pady=5)

        # Preprocessing Options
        ttk.Label(self.pre_frame, text="Preprocessing:", width=label_width, anchor="w").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.pre_mask_clouds_var = tk.BooleanVar()
        self.pre_mask_water_var = tk.BooleanVar()
        self.pre_mask_cloud_shadows_var = tk.BooleanVar()

        self.pre_cloud_checkbox = ttk.Checkbutton(self.pre_frame, text="Mask Clouds", variable=self.pre_mask_clouds_var, command=self.pre_toggle_preprocessing_path)
        self.pre_cloud_checkbox.grid(row=3, column=1, sticky="w", padx=(0, 20))

        self.pre_water_checkbox = ttk.Checkbutton(self.pre_frame, text="Mask Water", variable=self.pre_mask_water_var, command=self.pre_toggle_preprocessing_path)
        self.pre_water_checkbox.grid(row=3, column=1, sticky="e")

        # Cloud Shadows Checkbox (Initially Hidden)
        self.pre_cloud_shadows_checkbox = ttk.Checkbutton(self.pre_frame, text="Mask Cloud Shadows", variable=self.pre_mask_cloud_shadows_var, command=self.pre_update_cloud_methods)
        self.pre_cloud_shadows_checkbox.grid(row=4, column=1, sticky="w", padx=(0, 20))
        self.pre_cloud_shadows_checkbox.grid_remove()

        # NDWI Threshold Input (Initially Hidden)
        self.pre_ndwi_label = ttk.Label(self.pre_frame, text="NDWI Threshold:", width=label_width, anchor="e")
        self.pre_ndwi_entry = ttk.Entry(self.pre_frame, width=7)
        self.pre_ndwi_entry.insert(0, "0.01")  # Default value
        self.pre_ndwi_label.grid(row=4, column=1, padx=0, pady=5, sticky="e")
        self.pre_ndwi_entry.grid(row=4, column=2, padx=0, pady=5, sticky="w")
        self.pre_ndwi_label.grid_remove()
        self.pre_ndwi_entry.grid_remove()

        # Cloud Masking Method (Initially Hidden)
        self.pre_cloud_method_label = ttk.Label(self.pre_frame, text="Cloud Masking Method:", width=label_width, anchor="w")
        self.pre_cloud_method_var = tk.StringVar(value="standard")
        self.pre_cloud_method_dropdown = ttk.Combobox(self.pre_frame, textvariable=self.pre_cloud_method_var, state="readonly", width=15)
        self.pre_cloud_method_dropdown.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.pre_cloud_method_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.pre_cloud_method_dropdown.bind("<<ComboboxSelected>>", self.pre_update_warning)
        self.pre_cloud_method_dropdown.grid_remove()
        self.pre_cloud_method_label.grid_remove()

        # Warning Label (Initially Hidden)
        self.pre_warning_label = ttk.Label(self.pre_frame, text="This method may take a long time!", foreground="red", )
        self.pre_warning_label.grid(row=5, column=1, padx=0, pady=5, sticky="e")
        self.pre_warning_label.grid_remove()  # Hide it by default

        self.pre_update_cloud_methods()

        # Save Preprocessed Image Path (Initially Hidden)
        self.pre_preprocess_label = ttk.Label(self.pre_frame, text="Save Prep. Image As:", width=label_width, anchor="w")
        self.pre_preprocess_entry = ttk.Entry(self.pre_frame, width=50)
        self.pre_preprocess_button = ttk.Button(self.pre_frame, text="...", command=self.pre_select_preprocessed_path, width=3)

        self.pre_preprocess_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.pre_preprocess_entry.grid(row=6, column=1, padx=5, pady=5)
        self.pre_preprocess_button.grid(row=6, column=2, padx=5, pady=5)
        self.pre_preprocess_label.grid_remove()
        self.pre_preprocess_entry.grid_remove()
        self.pre_preprocess_button.grid_remove()

        # Frame for After Image Processing
        self.post_frame = ttk.LabelFrame(root, text="After Image", padding=(10, 5, 10, 5))
        self.post_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        # Post Image Selection
        ttk.Label(self.post_frame, text="Select After Image:", width=label_width, anchor="w").grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.post_entry = ttk.Entry(self.post_frame, width=50)
        self.post_entry.grid(row=7, column=1, padx=5, pady=5)
        self.post_button = ttk.Button(self.post_frame, text="...", command=self.load_post_image, width=3)
        self.post_button.grid(row=7, column=2, padx=5, pady=5)
        self.post_folder_button = ttk.Button(self.post_frame, text="📁", command=self.load_post_folder, width=3)
        self.post_folder_button.grid(row=7, column=3, padx=5, pady=5)

        # Post Image NoData
        ttk.Label(self.post_frame, text="Nodata value:", width=label_width, anchor="w").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.post_nodata_entry = ttk.Entry(self.post_frame, width=7)
        self.post_nodata_entry.grid(row=8, column=1, padx=5, pady=5, sticky="w")

        # Post Image Preview
        self.post_preview_button = ttk.Button(self.post_frame, text="👁️", command=self.post_preview, width=3)
        self.post_preview_button.grid(row=7, column=4, padx=5, pady=5)

        # Preprocessing Options
        ttk.Label(self.post_frame, text="Preprocessing:", width=label_width, anchor="w").grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.post_mask_clouds_var = tk.BooleanVar()
        self.post_mask_water_var = tk.BooleanVar()
        self.post_mask_cloud_shadows_var = tk.BooleanVar()

        self.post_cloud_checkbox = ttk.Checkbutton(self.post_frame, text="Mask Clouds", variable=self.post_mask_clouds_var, command=self.post_toggle_preprocessing_path)
        self.post_cloud_checkbox.grid(row=9, column=1, sticky="w", padx=(0, 20))

        self.post_water_checkbox = ttk.Checkbutton(self.post_frame, text="Mask Water", variable=self.post_mask_water_var, command=self.post_toggle_preprocessing_path)
        self.post_water_checkbox.grid(row=9, column=1, sticky="e")

        # Cloud Shadows Checkbox (Initially Hidden)
        self.post_cloud_shadows_checkbox = ttk.Checkbutton(self.post_frame, text="Mask Cloud Shadows", variable=self.post_mask_cloud_shadows_var, command=self.post_update_cloud_methods)
        self.post_cloud_shadows_checkbox.grid(row=10, column=1, sticky="w", padx=(0, 20))
        self.post_cloud_shadows_checkbox.grid_remove()

        # NDWI Threshold Input (Initially Hidden)
        self.post_ndwi_label = ttk.Label(self.post_frame, text="NDWI Threshold:", width=label_width, anchor="e")
        self.post_ndwi_entry = ttk.Entry(self.post_frame, width=7)
        self.post_ndwi_entry.insert(0, "0.01")  # Default value
        self.post_ndwi_label.grid(row=10, column=1, padx=0, pady=5, sticky="e")
        self.post_ndwi_entry.grid(row=10, column=2, padx=0, pady=5, sticky="w")
        self.post_ndwi_label.grid_remove()
        self.post_ndwi_entry.grid_remove()

        # Cloud Masking Method (Initially Hidden)
        self.post_cloud_method_label = ttk.Label(self.post_frame, text="Cloud Masking Method:", width=label_width, anchor="w")
        self.post_cloud_method_var = tk.StringVar(value="standard")
        self.post_cloud_method_dropdown = ttk.Combobox(self.post_frame, textvariable=self.post_cloud_method_var, state="readonly", width=15)
        self.post_cloud_method_dropdown.grid(row=11, column=1, padx=5, pady=5, sticky="w")
        self.post_cloud_method_label.grid(row=11, column=0, padx=5, pady=5, sticky="w")
        self.post_cloud_method_dropdown.bind("<<ComboboxSelected>>", self.post_update_warning)
        self.post_cloud_method_dropdown.grid_remove()
        self.post_cloud_method_label.grid_remove()

        # Warning Label (Initially Hidden)
        self.post_warning_label = ttk.Label(self.post_frame, text="This method may take a long time!", foreground="red", )
        self.post_warning_label.grid(row=11, column=1, padx=0, pady=5, sticky="e")
        self.post_warning_label.grid_remove()  # Hide it by default

        self.post_update_cloud_methods()

        # Save Preprocessed Image Path (Initially Hidden)
        self.post_preprocess_label = ttk.Label(self.post_frame, text="Save Prep. Image As:", width=label_width, anchor="w")
        self.post_preprocess_entry = ttk.Entry(self.post_frame, width=50)
        self.post_preprocess_button = ttk.Button(self.post_frame, text="...", command=self.post_select_preprocessed_path, width=3)

        self.post_preprocess_label.grid(row=12, column=0, padx=5, pady=5, sticky="w")
        self.post_preprocess_entry.grid(row=12, column=1, padx=5, pady=5)
        self.post_preprocess_button.grid(row=12, column=2, padx=5, pady=5)
        self.post_preprocess_label.grid_remove()
        self.post_preprocess_entry.grid_remove()
        self.post_preprocess_button.grid_remove()
        
        # Model Options
        self.model_frame = ttk.LabelFrame(root, text="Model Settings", padding=(10, 5, 10, 5))
        self.model_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Model Training Method Selection
        self.model_method_label = ttk.Label(self.model_frame, text="Select Model Training Method:")
        self.model_method_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Radiobuttons for Method Selection
        self.model_method_var = tk.StringVar(value="threshold")  # Default selection

        self.threshold_radiobutton = ttk.Radiobutton(
            self.model_frame, text="Threshold", variable=self.model_method_var, value="threshold", command=self.toggle_threshold_inputs
        )
        self.threshold_radiobutton.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.extreme_values_radiobutton = ttk.Radiobutton(
            self.model_frame, text="Extreme Values", variable=self.model_method_var, value="extreme", command=self.toggle_threshold_inputs
        )
        self.extreme_values_radiobutton.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Input Fields (Initially Showing Threshold Input)
        self.threshold_label = ttk.Label(self.model_frame, text="dNBR Threshold Value:")
        self.threshold_entry = ttk.Entry(self.model_frame, width=10)
        self.threshold_entry.insert(0, "300")  # Default value

        self.extreme_label = ttk.Label(self.model_frame, text="Extreme dNBR Thresholds:")
        self.extreme_unburned_entry = ttk.Entry(self.model_frame, width=10)
        self.extreme_unburned_entry.insert(0, "50")  # Default Min value
        self.extreme_burned_entry = ttk.Entry(self.model_frame, width=10)
        self.extreme_burned_entry.insert(0, "500")  # Default Max value

        # Display dNBR checkbox
        self.display_dnbr_var = tk.BooleanVar(value=False)  # Default: Show dNBR
        self.display_dnbr_checkbox = ttk.Checkbutton(self.model_frame, text="Display dNBR", variable=self.display_dnbr_var)
        self.display_dnbr_checkbox.grid(row=1, column=3)

        # Number of Samples & Train-Test Split
        ttk.Label(self.model_frame, text="Number of Samples:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.samples_entry = ttk.Entry(self.model_frame, width=10)
        self.samples_entry.insert(0, "5000")
        self.samples_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Train-Test Split with Spinbox (0 to 0.75)
        ttk.Label(self.model_frame, text="Train-Test Split:").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.split_var = tk.DoubleVar(value=0.75)  # Default value
        self.split_spinbox = ttk.Spinbox(
            self.model_frame, from_=0, to=1, increment=0.01, textvariable=self.split_var, width=5
        )
        self.split_spinbox.grid(row=2, column=3, padx=5, pady=5, sticky="w")

        # Save Model Checkbox
        self.save_model_var = tk.BooleanVar()
        self.save_model_checkbox = ttk.Checkbutton(
            self.model_frame, text="Save Model", variable=self.save_model_var, command=self.toggle_save_model_path
        )
        self.save_model_checkbox.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        # Save Model Path Entry (Initially Hidden)
        self.save_model_entry = ttk.Entry(self.model_frame, width=40)
        self.save_model_button = ttk.Button(self.model_frame, text="...", command=self.select_model_path, width=3)

        # Initialize the UI to show the correct inputs
        self.toggle_threshold_inputs()

        # Running
        self.running_frame = ttk.Frame(root, padding=(10, 5, 10, 5))
        self.running_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Classified Image Output Path
        tk.Label(self.running_frame, text="Save Classified Image As:", width=label_width, anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.output_entry = ttk.Entry(self.running_frame, width=50)
        self.output_entry.grid(row=0, column=1, padx=5, pady=5)
        self.output_button = ttk.Button(self.running_frame, text="...", command=self.select_output_path, width=3)
        self.output_button.grid(row=0, column=2, padx=5, pady=5)

        # Classify Button
        self.classify_button = ttk.Button(self.running_frame, text="Train and Classify", command=self.run_training_and_classification, width=20)
        self.classify_button.grid(row=1, column=1, pady=5)
        
        #Status Label
        self.status_label = ttk.Label(self.running_frame, text="", foreground="blue")  # Status message
        self.status_label.grid(row=2, column=1, pady=5)  # Place under the classify button

        # Go Back Button 
        self.back_button = ttk.Button(root, text="<-", command=self.go_back, width=5)
        self.back_button.grid(row=5, column=0, padx=10, pady=1, sticky="sw")  # Bottom left placement
        


    

    def pre_toggle_preprocessing_path(self):
        if self.pre_mask_clouds_var.get():
            self.pre_cloud_shadows_checkbox.grid()
            self.pre_cloud_method_dropdown.grid()
            self.pre_cloud_method_label.grid()
        else:
            self.pre_cloud_shadows_checkbox.grid_remove()
            self.pre_cloud_method_dropdown.grid_remove()
            self.pre_cloud_method_label.grid_remove()

        if self.pre_mask_water_var.get():
            self.pre_ndwi_label.grid()
            self.pre_ndwi_entry.grid()
        else:
            self.pre_ndwi_label.grid_remove()
            self.pre_ndwi_entry.grid_remove()

        if self.pre_mask_clouds_var.get() or self.pre_mask_water_var.get():
            self.pre_preprocess_label.grid()
            self.pre_preprocess_entry.grid()
            self.pre_preprocess_button.grid()
        else:
            self.pre_preprocess_label.grid_remove()
            self.pre_preprocess_entry.grid_remove()
            self.pre_preprocess_button.grid_remove()

    def load_pre_folder(self):
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if not folder_path:
            return
        
        satellite = self.satellite_var.get()
        self.update_state_label("Generating Stacked Raster for Pre Image...")
        
        try:
            stacked_image_path = image_loader.load_image(folder_path, satellite)
            self.update_state_label("Stacked Raster for Pre Image Generated!")
            self.pre_entry.delete(0, tk.END)
            self.pre_entry.insert(0, stacked_image_path)
            # Detect NoData value if applicable
            detected_nodata = preprocessing.detect_nodata_from_path(stacked_image_path)
            self.pre_nodata_entry.delete(0, tk.END)
            if detected_nodata is not None:
                self.pre_nodata_entry.delete(0, tk.END)
                self.pre_nodata_entry.insert(0, str(detected_nodata))

        except Exception as e:
            self.update_state_label("Error generating stacked raster for Pre Image")
            messagebox.showerror("Error", f"Failed to stack raster for Pre Image: {e}")

    def load_pre_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Raster Files", "*.tif;*.tiff;*.img;*.jp2")])
        if file_path:
            self.pre_entry.delete(0, tk.END)
            self.pre_entry.insert(0, file_path)
            self.pre_nodata_entry.delete(0, tk.END)
            detected_nodata = preprocessing.detect_nodata_from_path(file_path)
            if detected_nodata is not None:
                self.pre_nodata_entry.delete(0, tk.END)
                self.pre_nodata_entry.insert(0, str(detected_nodata))

    def pre_preview(self):
        image_path = self.pre_entry.get().strip()
        nodata_value = self.pre_nodata_entry.get().strip()
        if not image_path:
            messagebox.showerror("Error", "Please select an image file.")
            return
        # Call the function to display the preview
        self.display_rgb_image(image_path, nodata_value if nodata_value else None)

    def pre_select_preprocessed_path(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF Files", "*.tif")])
        if file_path:
            self.pre_preprocess_entry.delete(0, tk.END)
            self.pre_preprocess_entry.insert(0, file_path)

    def update_cloud_methods(self, event=None):
        self.pre_update_cloud_methods()
        self.post_update_cloud_methods()

    def pre_update_cloud_methods(self, event=None):
        selected_satellite = self.satellite_var.get()
        mask_shadows = self.pre_mask_cloud_shadows_var.get()
        
        if selected_satellite == "Sentinel-2":
            methods = ["standard", "auto", "scl", "omnicloudmask"] if mask_shadows else ["standard", "auto", "probability", "scl", "qa", "omnicloudmask"]
        elif selected_satellite == "Landsat 8/9":
            methods = ["qa", "auto", "omnicloudmask"]
        else:
            methods = []
        
        self.pre_cloud_method_dropdown["values"] = methods
        self.pre_cloud_method_var.set(methods[0])
        self.pre_update_warning()

    def pre_update_warning(self, event=None):
        if self.pre_cloud_method_var.get() in ["omnicloudmask", "auto"]:
            self.pre_warning_label.grid()
        else:
            self.pre_warning_label.grid_remove()

    def post_toggle_preprocessing_path(self):
        if self.post_mask_clouds_var.get():
            self.post_cloud_shadows_checkbox.grid()
            self.post_cloud_method_dropdown.grid()
            self.post_cloud_method_label.grid()
        else:
            self.post_cloud_shadows_checkbox.grid_remove()
            self.post_cloud_method_dropdown.grid_remove()
            self.post_cloud_method_label.grid_remove()

        if self.post_mask_water_var.get():
            self.post_ndwi_label.grid()
            self.post_ndwi_entry.grid()
        else:
            self.post_ndwi_label.grid_remove()
            self.post_ndwi_entry.grid_remove()

        if self.post_mask_clouds_var.get() or self.post_mask_water_var.get():
            self.post_preprocess_label.grid()
            self.post_preprocess_entry.grid()
            self.post_preprocess_button.grid()
        else:
            self.post_preprocess_label.grid_remove()
            self.post_preprocess_entry.grid_remove()
            self.post_preprocess_button.grid_remove()

    def load_post_folder(self):
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if not folder_path:
            return
        
        satellite = self.satellite_var.get()
        self.update_state_label("Generating Stacked Raster for Post Image...")
        
        try:
            stacked_image_path = image_loader.load_image(folder_path, satellite)
            self.update_state_label("Stacked Raster for Post Image Generated!")
            self.post_entry.delete(0, tk.END)
            self.post_entry.insert(0, stacked_image_path)
            # Detect NoData value if applicable
            detected_nodata = preprocessing.detect_nodata_from_path(stacked_image_path)
            self.post_nodata_entry.delete(0, tk.END)
            if detected_nodata is not None:
                self.post_nodata_entry.delete(0, tk.END)
                self.post_nodata_entry.insert(0, str(detected_nodata))

        except Exception as e:
            self.update_state_label("Error generating stacked raster for Post Image")
            messagebox.showerror("Error", f"Failed to stack raster for Post Image: {e}")

    def load_post_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Raster Files", "*.tif;*.tiff;*.img;*.jp2")])
        if file_path:
            self.post_entry.delete(0, tk.END)
            self.post_entry.insert(0, file_path)
            self.post_nodata_entry.delete(0, tk.END)
            detected_nodata = preprocessing.detect_nodata_from_path(file_path)
            if detected_nodata is not None:
                self.post_nodata_entry.delete(0, tk.END)
                self.post_nodata_entry.insert(0, str(detected_nodata))

    def post_preview(self):
        image_path = self.post_entry.get().strip()
        nodata_value = self.post_nodata_entry.get().strip()
        if not image_path:
            messagebox.showerror("Error", "Please select an image file.")
            return
        # Call the function to display the preview
        self.display_rgb_image(image_path, nodata_value if nodata_value else None)

    def post_select_preprocessed_path(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF Files", "*.tif")])
        if file_path:
            self.post_preprocess_entry.delete(0, tk.END)
            self.post_preprocess_entry.insert(0, file_path)

    def post_update_cloud_methods(self, event=None):
        selected_satellite = self.satellite_var.get()
        mask_shadows = self.post_mask_cloud_shadows_var.get()
        
        if selected_satellite == "Sentinel-2":
            methods = ["standard", "auto", "scl", "omnicloudmask"] if mask_shadows else ["standard", "auto", "probability", "scl", "qa", "omnicloudmask"]
        elif selected_satellite == "Landsat 8/9":
            methods = ["qa", "auto", "omnicloudmask"]
        else:
            methods = []
        
        self.post_cloud_method_dropdown["values"] = methods
        self.post_cloud_method_var.set(methods[0])
        self.post_update_warning()

    def post_update_warning(self, event=None):
        if self.post_cloud_method_var.get() in ["omnicloudmask", "auto"]:
            self.post_warning_label.grid()
        else:
            self.post_warning_label.grid_remove()

    def toggle_threshold_inputs(self):
        """Shows the appropriate input fields based on selected model method."""
        selected_method = self.model_method_var.get()

        if selected_method == "threshold":
            # Show Threshold input, Hide Extreme Values
            self.threshold_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.threshold_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

            self.extreme_label.grid_remove()
            self.extreme_unburned_entry.grid_remove()
            self.extreme_burned_entry.grid_remove()

        elif selected_method == "extreme":
            # Show Extreme Values input, Hide Threshold
            self.extreme_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.extreme_unburned_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            self.extreme_burned_entry.grid(row=1, column=2, padx=5, pady=5, sticky="w")

            self.threshold_label.grid_remove()
            self.threshold_entry.grid_remove()
    
    def toggle_save_model_path(self):
        """Shows or hides the save model path entry based on checkbox selection."""
        if self.save_model_var.get():
            self.save_model_entry.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
            self.save_model_button.grid(row=3, column=3, padx=5, pady=5, sticky="w")
        else:
            self.save_model_entry.grid_remove()
            self.save_model_button.grid_remove()
    
    def select_model_path(self):
        """Opens file dialog to select save path for the model."""
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            self.save_model_entry.delete(0, tk.END)
            self.save_model_entry.insert(0, file_path)

    def select_output_path(self):
        """Open file dialog to specify where to save the classified image"""
        file_path = filedialog.asksaveasfilename(defaultextension=".tif",
                                                 filetypes=[("TIFF Files", "*.tif")])
        if file_path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, file_path)

    def go_back(self):
        """Destroy the current UI and go back to the main menu"""
        for widget in self.root.winfo_children():
            widget.destroy()  # Clear the window
        BurnAreaClassifierApp(self.root)  # Reinitialize the main menu

    def update_state_label(self, content=""):
        self.status_label.config(text=content)
        self.root.update_idletasks()  # Refresh UI immediately

    def run_training_and_classification(self):
        """Starts the full training and classification process."""

        # Get input parameters from UI
        self.pre_image_path = self.pre_entry.get()
        self.post_image_path = self.post_entry.get()
        try:
            self.pre_nodata = float(self.pre_nodata_entry.get()) if self.pre_nodata_entry.get().strip() else None
        except ValueError:
            self.pre_nodata = None  # Defaults to None if input is invalid
        try:
            self.post_nodata = float(self.post_nodata_entry.get()) if self.post_nodata_entry.get().strip() else None
        except ValueError:
            self.post_nodata = None  # Defaults to None if input is invalid
        self.samples = int(self.samples_entry.get()) if self.samples_entry.get().isdigit() else 5000
        self.test_size = 1.0 - float(self.split_var.get()) 
        self.save_model_flag = self.save_model_var.get()
        self.model_save_path = self.save_model_entry.get() if self.save_model_entry.winfo_ismapped() else None
        self.pre_preprocess_path = self.pre_preprocess_entry.get() if self.pre_preprocess_entry.winfo_ismapped() else None
        self.post_preprocess_path = self.post_preprocess_entry.get() if self.post_preprocess_entry.winfo_ismapped() else None

        satellite_mapping = {
            "Sentinel-2": "S2",
            "Landsat 8/9": "L8"
        }

        # Get selected value from dropdown
        self.selected_satellite = self.satellite_var.get()
        self.satellite = satellite_mapping.get(self.selected_satellite, "S2")  # Default to "S2" if not found
        self.output_path = self.output_entry.get()



        self.method = self.model_method_var.get()
        self.threshold = int(self.threshold_entry.get()) if self.threshold_entry.get().isdigit() else 300
        self.ext_burned_threshold = int(self.extreme_burned_entry.get()) if self.extreme_burned_entry.get().isdigit() else 500
        self.ext_unburned_threshold = int(self.extreme_unburned_entry.get()) if self.extreme_unburned_entry.get().isdigit() else 50

        if not self.pre_image_path or not self.post_image_path:
            messagebox.showerror("Error", "Please select both Before and After images.")
            return
        if self.save_model_flag and not self.model_save_path:
            messagebox.showerror("Error", "Please select a path to save the model.")
            return
        
        self.update_state_label("Applying Masks...")

        # Create and start threads for preprocessing
        self.pre_thread = threading.Thread(target=self.preprocess_image, args=(self.pre_image_path, self.pre_nodata, "pre"))
        self.post_thread = threading.Thread(target=self.preprocess_image, args=(self.post_image_path, self.post_nodata, "post"))

        self.pre_thread.start()
        self.post_thread.start()

        # Schedule a check to see if both threads are done
        self.root.after(100, self.check_preprocessing_done)

    def preprocess_image(self, image_path, nodata, image_type):
        """Preprocesses the Before and After images."""

        apply_mask = (
            self.pre_mask_clouds_var.get() if image_type == "pre" else self.post_mask_clouds_var.get()
        ) or (
            self.pre_mask_water_var.get() if image_type == "pre" else self.post_mask_water_var.get()
        )

        if apply_mask:        
            try:

                preprocess_path = self.pre_preprocess_path if image_type == "pre" else self.post_preprocess_path

                processed_path = preprocessing.apply_masks(
                    image_path, self.satellite, 
                    preprocess_path if preprocess_path else None,
                    self.pre_cloud_method_var.get() if image_type == "pre" else self.post_cloud_method_var.get(),
                    self.pre_mask_clouds_var.get() if image_type == "pre" else self.post_mask_clouds_var.get(),
                    self.pre_mask_cloud_shadows_var.get() if image_type == "pre" else self.post_mask_cloud_shadows_var.get(),
                    self.pre_mask_water_var.get() if image_type == "pre" else self.post_mask_water_var.get(),
                    nodata
                )

                if image_type == "pre":
                    self.pre_image_path = processed_path
                else:
                    self.post_image_path = processed_path

            except Exception as e:
                messagebox.showerror(f"Preprocessing Error - {image_type.capitalize()} Image", f"Failed to preprocess {image_type} image: {e}")

    def check_preprocessing_done(self):
        """Checks if preprocessing threads have finished and moves to the next step."""
        if self.pre_thread.is_alive() or self.post_thread.is_alive():
            # If either thread is still running, check again in 100ms
            self.root.after(100, self.check_preprocessing_done)
        else:
            # Both threads have finished, move to the next step
            if not self.pre_image_path:
                    self.update_state_label("Masking Error on Pre Fire Image!")
                    messagebox.showerror("Masking Error", "An error happened while applying the masks to Pre Image. \nPlease make sure the needed bands for the selected method is present")
                    return
            if not self.post_image_path:
                    self.update_state_label("Masking Error on Post Fire Image!")
                    messagebox.showerror("Masking Error", "An error happened while applying the masks to Post Image. \nPlease make sure the needed bands for the selected method is present")
                    return
            self.update_state_label("Computing NBR and dNBR...")
            self.root.after(100, self.compute_nbr_and_dNBR)

    def compute_nbr_and_dNBR(self):
        """Computes NBR and dNBR, then moves to training."""
        try:
            pre_NBR = preprocessing.compute_NBR(self.pre_image_path, self.satellite, self.pre_nodata)
            post_NBR = preprocessing.compute_NBR(self.post_image_path, self.satellite, self.post_nodata)
            self.dNBR = processing.compute_dNBR(pre_NBR, post_NBR)


            if self.display_dnbr_var.get():
                self.update_state_label("Displaying dNBR...")
                self.root.after(100, self.display_dNBR)

            else:
                self.update_state_label("Creating Training Set...")
                self.root.after(100, self.create_training_set)
        
        except Exception as e:
            self.update_state_label("Error computing NBR")
            messagebox.showerror("Error", f"Failed to compute NBR: {e}")

    def display_dNBR(self):
        try:
            img_height, img_width = self.dNBR.shape
            aspect_ratio = img_width / img_height
            base_size = 6  # Base size for the figure
            fig = plt.figure(figsize=(base_size * aspect_ratio, base_size) if aspect_ratio > 1 else (base_size, base_size / aspect_ratio))

            cmap = mcolors.LinearSegmentedColormap.from_list("dNBR", ['darkgreen', 'green', 'yellow', 'orange', 'red', 'purple'])
            norm = mcolors.Normalize(vmin=-500, vmax=1300)
            plt.gca().set_aspect(self.dNBR.shape[1] / self.dNBR.shape[0])
            im = plt.imshow(self.dNBR, cmap=cmap, norm=norm) 
            cbar = plt.colorbar(im, orientation="horizontal", fraction=0.03, pad=0.1)
            cbar.set_label("dNBR Value")
            fig.canvas.manager.set_window_title("dNBR")  # Set window title
            plt.title("Raw dNBR Output")
            plt.show()

        except Exception as e:
            self.update_state_label("Error displaying dNBR")
            messagebox.showerror("Error", f"Failed to display dNBR: {e}")
            
        finally:
            self.update_state_label("Creating Training Set...")
            self.root.after(100, self.create_training_set)

    def create_training_set(self):
        """Creates training data and moves to model training."""
        try:
            self.X_sampled, self.y_sampled = processing.create_training_set(
                self.post_image_path, self.dNBR, method=self.method, threshold=self.threshold,
                ext_burned_threshold=self.ext_burned_threshold, ext_unburned_threshold=self.ext_unburned_threshold,
                samples=self.samples, post_image_nodata=self.post_nodata
            )

            self.update_state_label("Training Model...")
            self.root.after(100, self.train_model)

        except Exception as e:
            self.update_state_label("Error creating the training set!")
            messagebox.showerror("Error", f"Failed to create training set: {e}")

    def train_model(self):
        """Trains the model and evaluates it."""
        try:
            self.model, self.conf_matrix, self.class_report, self.accuracy = processing.train_and_evaluate_svm(
                self.X_sampled, self.y_sampled, test_size=self.test_size
            )

            # Save Model if Selected
            if self.save_model_flag and self.model_save_path:
                self.update_state_label("Saving Model...")
                self.root.after(100, self.save_trained_model)
            else:
                self.update_state_label("Training Complete! Displaying Metrics...")
                self.root.after(100, self.display_training_metrics)

        except Exception as e:
            self.update_state_label("Error during training!")
            messagebox.showerror("Error", f"Training failed: {e}")

    def save_trained_model(self):
        """Saves the trained model and moves to metrics display."""
        try:
            processing.save_model(self.model, self.model_save_path)
            self.update_state_label("Model Saved! Displaying Metrics...")
            self.root.after(100, self.display_training_metrics)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {e}")

    def display_training_metrics(self):
        """Displays training metrics in a structured, modern format using ttk."""
        
        # Check if class "1.0" exists, otherwise default to 0
        precision_0 = self.class_report.get("0.0", {}).get("precision", 0)
        recall_0 = self.class_report.get("0.0", {}).get("recall", 0)
        f1_score_0 = self.class_report.get("0.0", {}).get("f1-score", 0)
        support_0 = self.class_report.get("0.0", {}).get("support", 0)

        precision_1 = self.class_report.get("1.0", {}).get("precision", 0)
        recall_1 = self.class_report.get("1.0", {}).get("recall", 0)
        f1_score_1 = self.class_report.get("1.0", {}).get("f1-score", 0)
        support_1 = self.class_report.get("1.0", {}).get("support", 0)

        accuracy = self.class_report.get("accuracy", 0)

        # Create a new Toplevel window
        metrics_window = Toplevel(self.root)
        metrics_window.title("Training Metrics")
        metrics_window.geometry("500x350")  # Adjusted window size

        # Create a frame for layout using ttk
        frame = ttk.Frame(metrics_window, padding=10)
        frame.pack(fill="both", expand=True)

        # Title Label
        ttk.Label(frame, text="Model Training Report", font=("Arial", 14, "bold")).pack(pady=5)
        ttk.Label(frame, text=f"Overall Accuracy: {accuracy:.4f}", font=("Arial", 12, "bold")).pack()

        # Table for Classification Report
        table_frame = ttk.Frame(frame)
        table_frame.pack(pady=5)

        columns = ("Class", "Precision", "Recall", "F1-Score", "Support")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=3)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=80)

        # Insert rows for each class
        tree.insert("", "end", values=("Unburned (0)", f"{precision_0:.4f}", f"{recall_0:.4f}", f"{f1_score_0:.4f}", int(support_0)))
        tree.insert("", "end", values=("Burned (1)", f"{precision_1:.4f}", f"{recall_1:.4f}", f"{f1_score_1:.4f}", int(support_1)))

        tree.pack()

        # Confusion Matrix Section
        ttk.Label(frame, text="Confusion Matrix:", font=("Arial", 12, "bold")).pack()

        matrix_frame = ttk.Frame(frame, padding=5)
        matrix_frame.pack()

        # Confusion Matrix Table Headers
        labels = ["Actual 0", "Actual 1"]
        headers = ["Predicted 0", "Predicted 1"]

        for i, header in enumerate([""] + headers):
            ttk.Label(matrix_frame, text=header, font=("Arial", 10, "bold"), borderwidth=2, relief="ridge", width=12, anchor="center").grid(row=0, column=i, sticky="nsew", padx=2, pady=2)

        # Confusion Matrix Rows
        for i, label in enumerate(labels):
            ttk.Label(matrix_frame, text=label, font=("Arial", 10, "bold"), borderwidth=2, relief="ridge", width=12, anchor="center").grid(row=i+1, column=0, sticky="nsew", padx=2, pady=2)
            for j in range(2):
                ttk.Label(matrix_frame, text=str(self.conf_matrix[i][j]), font=("Arial", 10), borderwidth=2, relief="ridge", width=12, anchor="center").grid(row=i+1, column=j+1, sticky="nsew", padx=2, pady=2)

        # Close Button
        ttk.Button(frame, text="Close", command=metrics_window.destroy).pack(pady=10)

        self.update_state_label("Classifying Image...")
        self.root.after(100, self.classify_image)

    def classify_image(self):
        """Runs classification using the trained model."""
        try:
            self.classified_image_path = processing.classify(
                self.model, self.post_image_path, output_path = self.output_path if self.output_path else None,
                nodata_value=float(self.post_nodata) if self.post_nodata else None
            )

            self.update_state_label("Classification complete")

            # Ask if the user wants to preview
            open_preview = messagebox.askyesno(
                "Classification Complete",
                f"Classified image saved at {self.classified_image_path}\n\nDo you want to preview the image?"
            )

            if open_preview:
                self.root.after(100, lambda: self.display_classified_image(self.classified_image_path))

        except Exception as e:
            self.update_state_label("Classification failed")
            messagebox.showerror("Error", f"Classification failed: {e}")

    def display_rgb_image(self, image_path, nodata_value):
        """Displays an RGB preview using band descriptions to match Red, Green, and Blue bands."""
        try:
            with rio.open(image_path) as src:
                band_descriptions = src.descriptions  # Get band names

                if not band_descriptions or all(desc is None for desc in band_descriptions):
                    messagebox.showerror("Error", "No band descriptions found. Cannot determine RGB bands.")
                    return

                # Define band name variations for different satellites
                if self.satellite_var.get() == "Sentinel-2":
                    red_band_names = ["B4", "B04"]
                    green_band_names = ["B3", "B03"]
                    blue_band_names = ["B2", "B02"]
                elif self.satellite_var.get() == "Landsat 8/9":
                    red_band_names = ["B4", "SR_B4"]
                    green_band_names = ["B3", "SR_B3"]
                    blue_band_names = ["B2", "SR_B2"]
                else:
                    messagebox.showerror("Error", "Unknown satellite. Cannot determine RGB bands.")
                    return

                # Find the correct band indices dynamically
                band_indices = {"Red": None, "Green": None, "Blue": None}

                for i, desc in enumerate(band_descriptions):
                    if any(red in desc for red in red_band_names):
                        band_indices["Red"] = i + 1  # Rasterio uses 1-based indexing
                    elif any(green in desc for green in green_band_names):
                        band_indices["Green"] = i + 1
                    elif any(blue in desc for blue in blue_band_names):
                        band_indices["Blue"] = i + 1
                # Ensure all bands are found
                if None in band_indices.values():
                    messagebox.showerror("Error", "Could not find all RGB bands in the metadata.")
                    return

                # Read the bands
                r, g, b = band_indices["Red"], band_indices["Green"], band_indices["Blue"]
                red, green, blue = src.read(r).astype(np.float32), src.read(g).astype(np.float32), src.read(b).astype(np.float32)

                if nodata_value is not None and np.isnan(float(nodata_value)):
                    mask = np.isnan(red) | np.isnan(green) | np.isnan(blue)
                elif nodata_value is not None:
                    mask = (red == nodata_value) | (green == nodata_value) | (blue == nodata_value)
                else:
                    mask = np.zeros_like(red, dtype=bool)  # No NoData value in metadata, assume all valid
            
                # Normalize bands
                def normalize_band(band, mask):
                    band_min, band_max = np.min(band[~mask]), np.max(band[~mask])
                    normalized_band = (band - band_min) / (band_max - band_min)
                    normalized_band[mask] = 0  # Keep NoData areas black
                    return normalized_band

                red, green, blue = normalize_band(red, mask), normalize_band(green, mask), normalize_band(blue, mask)

                def gamma_correction(band, gamma=1.2):  # Default gamma 1.2 makes midtones brighter
                    return np.power(band, 1/gamma)

                red, green, blue = gamma_correction(red), gamma_correction(green), gamma_correction(blue)

                img_height, img_width = red.shape  # Get image size
                aspect_ratio = img_width / img_height
                base_size = 6  # Base figure size

                # Stack RGB and display
                rgb_image = np.dstack((red, green, blue))
                rgb_image[mask] = [1, 1, 1]  # White color for NoData area
                fig, ax = plt.subplots(figsize=(base_size * aspect_ratio, base_size) if aspect_ratio > 1 else (base_size, base_size / aspect_ratio))
                fig.canvas.manager.set_window_title("Image Preview")
                ax.imshow(rgb_image)
                ax.set_title("RGB Composite Preview")
                plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Could not display RGB preview: {e}")

    def display_classified_image(self, image_path):
        """Displays the classified image with a custom title and legend."""
        
        

        with rio.open(image_path) as src:
            classified_data = src.read(1)  # Read the first band

            img_height, img_width = classified_data.shape  # Get image size
            aspect_ratio = img_width / img_height
            base_size = 6  # Base figure size

            fig = plt.figure(figsize=(base_size * aspect_ratio, base_size) if aspect_ratio > 1 else (base_size, base_size / aspect_ratio))  # Create a figure
            fig.canvas.manager.set_window_title("Classified Image")  # Set window title
            

            ax = fig.add_subplot(111)  # Add subplot
            cmap = mcolors.ListedColormap(["green", "red"])  # 0 = Green (Unburned), 1 = Red (Burned)
            bounds = [0, 0.5, 1]  # Define boundaries for values
            norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Normalize colors to values

            img = ax.imshow(classified_data, cmap=cmap, norm=norm)
            ax.set_title("Classified Image")

            # Add legend
            legend_labels = [
                mpatches.Patch(color="green", label="Unburned (0)"),
                mpatches.Patch(color="red", label="Burned (1)")
            ]
            ax.legend(handles=legend_labels, loc="upper right")

        plt.show()  # Display the figure

class UseExistingModelUI:
    def __init__(self, root):
        """Use an existing model to classify"""
        self.root = root
        self.root.title("Use Existing Model")
        self.root.geometry("600x450")
        root.resizable(False, False)


        label_width = 22  # Set a fixed width for labels to avoid shifting

        # Satellite selection
        ttk.Label(root, text="Select Satellite:", width=label_width, anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.satellite_var = tk.StringVar(value="Sentinel-2")  # Default selection
        self.satellite_dropdown = ttk.Combobox(root, textvariable=self.satellite_var, values=["Sentinel-2", "Landsat 8/9"], state="readonly", width=15)
        self.satellite_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.satellite_dropdown.bind("<<ComboboxSelected>>", self.update_cloud_methods)

        # Model File Selection
        ttk.Label(root, text="Select Model File (.pkl):", width=label_width, anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.model_entry = ttk.Entry(root, width=50)
        self.model_entry.grid(row=1, column=1, padx=5, pady=5)
        self.model_button = ttk.Button(root, text="...", command=self.load_model_path, width=3)
        self.model_button.grid(row=1, column=2, padx=5, pady=5)

        # After Image Selection
        ttk.Label(root, text="Select After Image:", width=label_width, anchor="w").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.after_entry = ttk.Entry(root, width=50)
        self.after_entry.grid(row=2, column=1, padx=5, pady=5)
        self.after_button = ttk.Button(root, text="...", command=self.load_after_image, width=3)
        self.after_button.grid(row=2, column=2, padx=5, pady=5)
        self.post_folder_button = ttk.Button(root, text="📁", command=self.load_after_folder, width=3)
        self.post_folder_button.grid(row=2, column=3, padx=5, pady=5)

        # After Image NoData
        ttk.Label(root, text="Nodata value:", width=label_width, anchor="w").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.after_nodata_entry = ttk.Entry(root, width=7)
        self.after_nodata_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # After Image Preview
        self.after_preview_button = ttk.Button(root, text="👁️", command=self.after_preview, width=3)
        self.after_preview_button.grid(row=2, column=4, padx=5, pady=5)

        # Preprocessing Options
        ttk.Label(root, text="Preprocessing:", width=label_width, anchor="w").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.mask_clouds_var = tk.BooleanVar()
        self.mask_water_var = tk.BooleanVar()
        self.mask_cloud_shadows_var = tk.BooleanVar()

        self.cloud_checkbox = ttk.Checkbutton(root, text="Mask Clouds", variable=self.mask_clouds_var, command=self.toggle_preprocessing_path)
        self.cloud_checkbox.grid(row=4, column=1, sticky="w", padx=(0, 20))

        self.water_checkbox = ttk.Checkbutton(root, text="Mask Water", variable=self.mask_water_var, command=self.toggle_preprocessing_path)
        self.water_checkbox.grid(row=4, column=1, sticky="e")

        # Cloud Shadows Checkbox (Initially Hidden)
        self.cloud_shadows_checkbox = ttk.Checkbutton(root, text="Mask Cloud Shadows", variable=self.mask_cloud_shadows_var, command=self.update_cloud_methods)
        self.cloud_shadows_checkbox.grid(row=5, column=1, sticky="w", padx=(0, 20))
        self.cloud_shadows_checkbox.grid_remove()

        # NDWI Threshold Input (Initially Hidden)
        self.ndwi_label = ttk.Label(root, text="NDWI Threshold:", width=label_width, anchor="e")
        self.ndwi_entry = ttk.Entry(root, width=7)
        self.ndwi_entry.insert(0, "0.01")  # Default value
        self.ndwi_label.grid(row=5, column=1, padx=0, pady=5, sticky="e")
        self.ndwi_entry.grid(row=5, column=2, padx=0, pady=5, sticky="w")
        self.ndwi_label.grid_remove()
        self.ndwi_entry.grid_remove()

        # Cloud Masking Method (Initially Hidden)
        self.cloud_method_label = ttk.Label(root, text="Cloud Masking Method:", width=label_width, anchor="w")
        self.cloud_method_var = tk.StringVar(value="standard")
        self.cloud_method_dropdown = ttk.Combobox(root, textvariable=self.cloud_method_var, state="readonly", width=15)
        self.cloud_method_dropdown.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        self.cloud_method_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.cloud_method_dropdown.bind("<<ComboboxSelected>>", self.update_warning)
        self.cloud_method_dropdown.grid_remove()
        self.cloud_method_label.grid_remove()

        # Warning Label (Initially Hidden)
        self.warning_label = ttk.Label(root, text="This method may take a long time!", foreground="red", )
        self.warning_label.grid(row=6, column=1, padx=0, pady=5, sticky="e")
        self.warning_label.grid_remove()  # Hide it by default

        self.update_cloud_methods()

        # Save Preprocessed Image Path (Initially Hidden)
        self.preprocess_label = ttk.Label(root, text="Save Prep. Image As:", width=label_width, anchor="w")
        self.preprocess_entry = ttk.Entry(root, width=50)
        self.preprocess_button = ttk.Button(root, text="...", command=self.select_preprocessed_path, width=3)

        self.preprocess_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.preprocess_entry.grid(row=7, column=1, padx=5, pady=5)
        self.preprocess_button.grid(row=7, column=2, padx=5, pady=5)
        self.preprocess_label.grid_remove()
        self.preprocess_entry.grid_remove()
        self.preprocess_button.grid_remove()

        # Classified Image Output Path
        tk.Label(root, text="Save Classified Image As:", width=label_width, anchor="w").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.output_entry = ttk.Entry(root, width=50)
        self.output_entry.grid(row=8, column=1, padx=5, pady=5)
        self.output_button = ttk.Button(root, text="...", command=self.select_output_path, width=3)
        self.output_button.grid(row=8, column=2, padx=5, pady=5)

        # Classify Button
        self.classify_button = ttk.Button(root, text="Classify Image", command=self.run_classification, width=20)
        self.classify_button.grid(row=9, column=1, pady=10)
        
        #Status Label
        self.status_label = ttk.Label(root, text="", foreground="blue")  # Status message
        self.status_label.grid(row=10, column=1, pady=5)  # Place under the classify button

        # Go Back Button 
        self.back_button = ttk.Button(root, text="<-", command=self.go_back, width=5)
        self.back_button.grid(row=11, column=0, padx=10, pady=20, sticky="sw")  # Bottom left placement

    def go_back(self):
        """Destroy the current UI and go back to the main menu"""
        for widget in self.root.winfo_children():
            widget.destroy()  # Clear the window
        BurnAreaClassifierApp(self.root)  # Reinitialize the main menu

    def toggle_preprocessing_path(self):
        """Show or hide elements based on user selection"""
        if self.mask_clouds_var.get():
            self.cloud_shadows_checkbox.grid()
            self.cloud_method_dropdown.grid()
            self.cloud_method_label.grid()
        else:
            self.cloud_shadows_checkbox.grid_remove()
            self.cloud_method_dropdown.grid_remove()
            self.cloud_method_label.grid_remove()

        if self.mask_water_var.get():
            self.ndwi_label.grid()
            self.ndwi_entry.grid()
        else:
            self.ndwi_label.grid_remove()
            self.ndwi_entry.grid_remove()

        if self.mask_clouds_var.get() or self.mask_water_var.get():
            self.preprocess_label.grid()
            self.preprocess_entry.grid()
            self.preprocess_button.grid()
        else:
            self.preprocess_label.grid_remove()
            self.preprocess_entry.grid_remove()
            self.preprocess_button.grid_remove()

    def update_cloud_methods(self, event=None):
        """Dynamically update the cloud masking method dropdown based on selections."""
        selected_satellite = self.satellite_var.get()
        mask_shadows = self.mask_cloud_shadows_var.get()

        # Define method options for different scenarios
        if selected_satellite == "Sentinel-2":
            if mask_shadows:
                methods = ["standard", "auto", "scl", "omnicloudmask"]  # Remove 'qa' and 'probability'
            elif not mask_shadows:
                methods = ["standard", "auto", "probability", "scl", "qa", "omnicloudmask"]
        elif selected_satellite == "Landsat 8/9":  # Landsat 8/9
            methods = ["qa", "auto", "omnicloudmask"]
        else:
            methods = []

        # Update dropdown values
        self.cloud_method_dropdown["values"] = methods
        self.cloud_method_var.set(methods[0])  # Reset to first valid option
        self.update_warning()

    def update_warning(self, event=None):
        # Show warning if slow methods are selected
        if self.cloud_method_var.get() in ["omnicloudmask", "auto"]:
            self.warning_label.grid()
        else:
            self.warning_label.grid_remove()

    def update_state_label(self, content=""):
        self.status_label.config(text=content)
        self.root.update_idletasks()  # Refresh UI immediately

    def load_model_path(self):
        """Open file dialog to select a model file"""
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, file_path)

    def load_after_folder(self):
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if not folder_path:
            return
        
        satellite = self.satellite_var.get()
        self.update_state_label("Generating Stacked Raster for After Image...")
        
        try:
            stacked_image_path = image_loader.load_image(folder_path, satellite)
            self.update_state_label("Stacked Raster for After Image Generated!")
            self.after_entry.delete(0, tk.END)
            self.after_entry.insert(0, stacked_image_path)
            # Detect NoData value if applicable
            detected_nodata = preprocessing.detect_nodata_from_path(stacked_image_path)
            self.after_nodata_entry.delete(0, tk.END)
            if detected_nodata is not None:
                self.after_nodata_entry.delete(0, tk.END)
                self.after_nodata_entry.insert(0, str(detected_nodata))

        except Exception as e:
            self.update_state_label("Error generating stacked raster for After Image")
            messagebox.showerror("Error", f"Failed to stack raster for After Image: {e}")

    def load_after_image(self):
        """Open file dialog to select after image"""
        file_path = filedialog.askopenfilename(filetypes=[("Raster Files", "*.tif;*.tiff;*.img;*.jp2")])
        if file_path:
            self.after_entry.delete(0, tk.END)
            self.after_entry.insert(0, file_path)
            # Detect NoData and populate the entry field
            detected_nodata = preprocessing.detect_nodata_from_path(file_path)
            if detected_nodata is not None:
                self.after_nodata_entry.delete(0, tk.END)
                self.after_nodata_entry.insert(0, str(detected_nodata))

    def after_preview(self):
        image_path = self.after_entry.get().strip()
        nodata_value = self.after_nodata_entry.get().strip()
        if not image_path:
            messagebox.showerror("Error", "Please select an image file.")
            return
        # Call the function to display the preview
        self.display_rgb_image(image_path, nodata_value if nodata_value else None)

    def select_preprocessed_path(self):
        """Open file dialog to specify where to save the preprocessed image"""
        file_path = filedialog.asksaveasfilename(defaultextension=".tif",
                                                 filetypes=[("TIFF Files", "*.tif")])
        if file_path:
            self.preprocess_entry.delete(0, tk.END)
            self.preprocess_entry.insert(0, file_path)

    def select_output_path(self):
        """Open file dialog to specify where to save the classified image"""
        file_path = filedialog.asksaveasfilename(defaultextension=".tif",
                                                 filetypes=[("TIFF Files", "*.tif")])
        if file_path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, file_path)
    
    def run_classification(self):
        """Run classification using an existing model"""
        self.model_path = self.model_entry.get()
        self.after_image_path = self.after_entry.get()
        self.after_nodata = self.after_nodata_entry.get()
        self.output_path = self.output_entry.get()
        self.mask_clouds = self.mask_clouds_var.get()
        self.mask_shadows = self.mask_cloud_shadows_var.get()
        self.mask_water = self.mask_water_var.get()
        self.preprocessed_path = self.preprocess_entry.get() if self.preprocess_entry.winfo_ismapped() else None

        satellite_mapping = {
            "Sentinel-2": "S2",
            "Landsat 8/9": "L8"
        }

        # Get selected value from dropdown
        self.selected_satellite = self.satellite_var.get()
        self.selected_method = self.cloud_method_var.get()
        self.satellite = satellite_mapping.get(self.selected_satellite, "S2")  # Default to "S2" if not found

        if not self.model_path or not self.after_image_path:
            messagebox.showerror("Error", "Please select a Model and an After image")
            return

        # Start with first step
        self.update_state_label("Applying masks...")
        self.root.after(100, self.apply_masks)  # Move to next step asynchronously

    def apply_masks(self):
        """Applies masks if needed"""
        if self.mask_clouds or self.mask_water:
            try:
                self.after_image_path = preprocessing.apply_masks(
                    self.after_image_path, self.satellite,
                    self.preprocessed_path if self.preprocessed_path else None,
                    self.selected_method, self.mask_clouds, self.mask_shadows,
                    self.mask_water, self.after_nodata if self.after_nodata else None
                )
                if not self.after_image_path:
                    self.update_state_label("Masking error!")
                    messagebox.showerror("Error", "An error happened while applying the masks. Please make sure the needed bands for the selected method is present")
                    return
            except Exception as e:
                self.update_state_label("Masking error!")
                messagebox.showerror("Error", f"Masking error: {e}")
                return

        # Proceed to loading model
        self.update_state_label("Loading model...")
        self.root.after(100, self.load_model)  # Move to next step asynchronously

    def load_model(self):
        """Loads the classification model"""
        try:
            self.model = processing.load_model(self.model_path)
        except Exception as e:
            self.update_state_label("Error loading model")
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        # Proceed to classification
        self.update_state_label("Classifying image...")
        self.root.after(100, self.classify_image)  # Move to next step asynchronously

    def classify_image(self):
        """Runs the classification process"""
        try:
            self.classified_image_path = processing.classify(
                self.model, self.after_image_path,
                output_path=self.output_path if self.output_path else None,
                nodata_value=float(self.after_nodata) if self.after_nodata else None
            )
            self.update_state_label("Classification complete")
            
            # Ask if the user wants to preview
            open_preview = messagebox.askyesno(
                "Classification Complete",
                f"Classified image saved at {self.classified_image_path}\n\nDo you want to preview the image?"
            )

            if open_preview:
                self.root.after(100, lambda: self.display_classified_image(self.classified_image_path))  # Open preview asynchronously

        except Exception as e:
            self.update_state_label("Classification failed")
            messagebox.showerror("Error", f"Classification failed: {e}")


    def display_classified_image(self, image_path):
        """Displays the classified image with a custom title and legend."""
        
        

        with rio.open(image_path) as src:
            classified_data = src.read(1)  # Read the first band

            img_height, img_width = classified_data.shape  # Get image size
            aspect_ratio = img_width / img_height
            base_size = 6  # Base figure size

            fig = plt.figure(figsize=(base_size * aspect_ratio, base_size) if aspect_ratio > 1 else (base_size, base_size / aspect_ratio))  # Create a figure
            fig.canvas.manager.set_window_title("Classified Image")  # Set window title
            

            ax = fig.add_subplot(111)  # Add subplot
            cmap = mcolors.ListedColormap(["green", "red"])  # 0 = Green (Unburned), 1 = Red (Burned)
            bounds = [0, 0.5, 1]  # Define boundaries for values
            norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Normalize colors to values

            img = ax.imshow(classified_data, cmap=cmap, norm=norm)
            ax.set_title("Classified Image")

            # Add legend
            legend_labels = [
                mpatches.Patch(color="green", label="Unburned (0)"),
                mpatches.Patch(color="red", label="Burned (1)")
            ]
            ax.legend(handles=legend_labels, loc="upper right")

        plt.show()  # Display the figure

    def display_rgb_image(self, image_path, nodata_value):
        """Displays an RGB preview using band descriptions to match Red, Green, and Blue bands."""
        try:
            with rio.open(image_path) as src:
                band_descriptions = src.descriptions  # Get band names

                if not band_descriptions or all(desc is None for desc in band_descriptions):
                    messagebox.showerror("Error", "No band descriptions found. Cannot determine RGB bands.")
                    return

                # Define band name variations for different satellites
                if self.satellite_var.get() == "Sentinel-2":
                    red_band_names = ["B4", "B04"]
                    green_band_names = ["B3", "B03"]
                    blue_band_names = ["B2", "B02"]
                elif self.satellite_var.get() == "Landsat 8/9":
                    red_band_names = ["B4", "SR_B4"]
                    green_band_names = ["B3", "SR_B3"]
                    blue_band_names = ["B2", "SR_B2"]
                else:
                    messagebox.showerror("Error", "Unknown satellite. Cannot determine RGB bands.")
                    return

                # Find the correct band indices dynamically
                band_indices = {"Red": None, "Green": None, "Blue": None}

                for i, desc in enumerate(band_descriptions):
                    if any(red in desc for red in red_band_names):
                        band_indices["Red"] = i + 1  # Rasterio uses 1-based indexing
                    elif any(green in desc for green in green_band_names):
                        band_indices["Green"] = i + 1
                    elif any(blue in desc for blue in blue_band_names):
                        band_indices["Blue"] = i + 1
                # Ensure all bands are found
                if None in band_indices.values():
                    messagebox.showerror("Error", "Could not find all RGB bands in the metadata.")
                    return

                # Read the bands
                r, g, b = band_indices["Red"], band_indices["Green"], band_indices["Blue"]
                red, green, blue = src.read(r).astype(np.float32), src.read(g).astype(np.float32), src.read(b).astype(np.float32)

                if nodata_value is not None and np.isnan(float(nodata_value)):
                    mask = np.isnan(red) | np.isnan(green) | np.isnan(blue)
                elif nodata_value is not None:
                    mask = (red == nodata_value) | (green == nodata_value) | (blue == nodata_value)
                else:
                    mask = np.zeros_like(red, dtype=bool)  # No NoData value in metadata, assume all valid
            
                # Normalize bands
                def normalize_band(band, mask):
                    band_min, band_max = np.min(band[~mask]), np.max(band[~mask])
                    normalized_band = (band - band_min) / (band_max - band_min)
                    normalized_band[mask] = 0  # Keep NoData areas black
                    return normalized_band

                red, green, blue = normalize_band(red, mask), normalize_band(green, mask), normalize_band(blue, mask)

                def gamma_correction(band, gamma=1.2):  # Default gamma 1.2 makes midtones brighter
                    return np.power(band, 1/gamma)

                red, green, blue = gamma_correction(red), gamma_correction(green), gamma_correction(blue)

                img_height, img_width = red.shape  # Get image size
                aspect_ratio = img_width / img_height
                base_size = 6  # Base figure size

                # Stack RGB and display
                rgb_image = np.dstack((red, green, blue))
                rgb_image[mask] = [1, 1, 1]  # White color for NoData area
                fig, ax = plt.subplots(figsize=(base_size * aspect_ratio, base_size) if aspect_ratio > 1 else (base_size, base_size / aspect_ratio))
                fig.canvas.manager.set_window_title("Image Preview")
                ax.imshow(rgb_image)
                ax.set_title("RGB Composite Preview")
                plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Could not display RGB preview: {e}")

class TestModelPerformanceUI:
    def __init__(self, root):
        """UI for model evaluation"""
        self.root = root
        self.root.title("Test Model Performance")
        self.root.geometry("650x850")
        root.resizable(False, False)
    
        
        label_width = 22
        
        # Satellite Selection
        ttk.Label(root, text="Select Satellite:", width=label_width, anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.satellite_var = tk.StringVar(value="Sentinel-2")
        self.satellite_dropdown = ttk.Combobox(root, textvariable=self.satellite_var, values=["Sentinel-2", "Landsat 8/9"], state="readonly", width=15)
        self.satellite_dropdown.grid(row=0, column=0, padx=100, pady=5, sticky="w")
        self.satellite_dropdown.bind("<<ComboboxSelected>>", self.update_cloud_methods)
        
        # Frame for Before Image Processing
        self.pre_frame = ttk.LabelFrame(root, text="Before Image", padding=(10, 5, 10, 5))
        self.pre_frame.grid(row=1, column=0, columnspan=5, padx=10, pady=10, sticky="ew")

        # Before Image Selection
        ttk.Label(self.pre_frame, text="Select Before Image:", width=label_width, anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pre_entry = ttk.Entry(self.pre_frame, width=50)
        self.pre_entry.grid(row=0, column=1, padx=5, pady=5)
        self.pre_button = ttk.Button(self.pre_frame, text="...", command=self.load_pre_image, width=3)
        self.pre_button.grid(row=0, column=2, padx=5, pady=5)
        self.pre_folder_button = ttk.Button(self.pre_frame, text="📁", command=self.load_pre_folder, width=3)
        self.pre_folder_button.grid(row=0, column=3, padx=5, pady=5)

        # Pre Image NoData
        ttk.Label(self.pre_frame, text="Nodata value:", width=label_width, anchor="w").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.pre_nodata_entry = ttk.Entry(self.pre_frame, width=7)
        self.pre_nodata_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Before Image Preview
        self.pre_preview_button = ttk.Button(self.pre_frame, text="👁️", command=self.pre_preview, width=3)
        self.pre_preview_button.grid(row=0, column=4, padx=5, pady=5)

        # Preprocessing Options
        ttk.Label(self.pre_frame, text="Preprocessing:", width=label_width, anchor="w").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.pre_mask_clouds_var = tk.BooleanVar()
        self.pre_mask_water_var = tk.BooleanVar()
        self.pre_mask_cloud_shadows_var = tk.BooleanVar()

        self.pre_cloud_checkbox = ttk.Checkbutton(self.pre_frame, text="Mask Clouds", variable=self.pre_mask_clouds_var, command=self.pre_toggle_preprocessing_path)
        self.pre_cloud_checkbox.grid(row=3, column=1, sticky="w", padx=(0, 20))

        self.pre_water_checkbox = ttk.Checkbutton(self.pre_frame, text="Mask Water", variable=self.pre_mask_water_var, command=self.pre_toggle_preprocessing_path)
        self.pre_water_checkbox.grid(row=3, column=1, sticky="e")

        # Cloud Shadows Checkbox (Initially Hidden)
        self.pre_cloud_shadows_checkbox = ttk.Checkbutton(self.pre_frame, text="Mask Cloud Shadows", variable=self.pre_mask_cloud_shadows_var, command=self.pre_update_cloud_methods)
        self.pre_cloud_shadows_checkbox.grid(row=4, column=1, sticky="w", padx=(0, 20))
        self.pre_cloud_shadows_checkbox.grid_remove()

        # NDWI Threshold Input (Initially Hidden)
        self.pre_ndwi_label = ttk.Label(self.pre_frame, text="NDWI Threshold:", width=label_width, anchor="e")
        self.pre_ndwi_entry = ttk.Entry(self.pre_frame, width=7)
        self.pre_ndwi_entry.insert(0, "0.01")  # Default value
        self.pre_ndwi_label.grid(row=4, column=1, padx=0, pady=5, sticky="e")
        self.pre_ndwi_entry.grid(row=4, column=2, padx=0, pady=5, sticky="w")
        self.pre_ndwi_label.grid_remove()
        self.pre_ndwi_entry.grid_remove()

        # Cloud Masking Method (Initially Hidden)
        self.pre_cloud_method_label = ttk.Label(self.pre_frame, text="Cloud Masking Method:", width=label_width, anchor="w")
        self.pre_cloud_method_var = tk.StringVar(value="standard")
        self.pre_cloud_method_dropdown = ttk.Combobox(self.pre_frame, textvariable=self.pre_cloud_method_var, state="readonly", width=15)
        self.pre_cloud_method_dropdown.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.pre_cloud_method_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.pre_cloud_method_dropdown.bind("<<ComboboxSelected>>", self.pre_update_warning)
        self.pre_cloud_method_dropdown.grid_remove()
        self.pre_cloud_method_label.grid_remove()

        # Warning Label (Initially Hidden)
        self.pre_warning_label = ttk.Label(self.pre_frame, text="This method may take a long time!", foreground="red", )
        self.pre_warning_label.grid(row=5, column=1, padx=0, pady=5, sticky="e")
        self.pre_warning_label.grid_remove()  # Hide it by default

        self.pre_update_cloud_methods()

        # Save Preprocessed Image Path (Initially Hidden)
        self.pre_preprocess_label = ttk.Label(self.pre_frame, text="Save Prep. Image As:", width=label_width, anchor="w")
        self.pre_preprocess_entry = ttk.Entry(self.pre_frame, width=50)
        self.pre_preprocess_button = ttk.Button(self.pre_frame, text="...", command=self.pre_select_preprocessed_path, width=3)

        self.pre_preprocess_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.pre_preprocess_entry.grid(row=6, column=1, padx=5, pady=5)
        self.pre_preprocess_button.grid(row=6, column=2, padx=5, pady=5)
        self.pre_preprocess_label.grid_remove()
        self.pre_preprocess_entry.grid_remove()
        self.pre_preprocess_button.grid_remove()

        # Frame for After Image Processing
        self.post_frame = ttk.LabelFrame(root, text="After Image", padding=(10, 5, 10, 5))
        self.post_frame.grid(row=2, column=0, columnspan=5, padx=10, pady=10, sticky="ew")

        # Post Image Selection
        ttk.Label(self.post_frame, text="Select After Image:", width=label_width, anchor="w").grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.post_entry = ttk.Entry(self.post_frame, width=50)
        self.post_entry.grid(row=7, column=1, padx=5, pady=5)
        self.post_button = ttk.Button(self.post_frame, text="...", command=self.load_post_image, width=3)
        self.post_button.grid(row=7, column=2, padx=5, pady=5)
        self.post_folder_button = ttk.Button(self.post_frame, text="📁", command=self.load_post_folder, width=3)
        self.post_folder_button.grid(row=7, column=3, padx=5, pady=5)

        # Post Image NoData
        ttk.Label(self.post_frame, text="Nodata value:", width=label_width, anchor="w").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.post_nodata_entry = ttk.Entry(self.post_frame, width=7)
        self.post_nodata_entry.grid(row=8, column=1, padx=5, pady=5, sticky="w")

        # Post Image Preview
        self.post_preview_button = ttk.Button(self.post_frame, text="👁️", command=self.post_preview, width=3)
        self.post_preview_button.grid(row=7, column=4, padx=5, pady=5)

        # Preprocessing Options
        ttk.Label(self.post_frame, text="Preprocessing:", width=label_width, anchor="w").grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.post_mask_clouds_var = tk.BooleanVar()
        self.post_mask_water_var = tk.BooleanVar()
        self.post_mask_cloud_shadows_var = tk.BooleanVar()

        self.post_cloud_checkbox = ttk.Checkbutton(self.post_frame, text="Mask Clouds", variable=self.post_mask_clouds_var, command=self.post_toggle_preprocessing_path)
        self.post_cloud_checkbox.grid(row=9, column=1, sticky="w", padx=(0, 20))

        self.post_water_checkbox = ttk.Checkbutton(self.post_frame, text="Mask Water", variable=self.post_mask_water_var, command=self.post_toggle_preprocessing_path)
        self.post_water_checkbox.grid(row=9, column=1, sticky="e")

        # Cloud Shadows Checkbox (Initially Hidden)
        self.post_cloud_shadows_checkbox = ttk.Checkbutton(self.post_frame, text="Mask Cloud Shadows", variable=self.post_mask_cloud_shadows_var, command=self.post_update_cloud_methods)
        self.post_cloud_shadows_checkbox.grid(row=10, column=1, sticky="w", padx=(0, 20))
        self.post_cloud_shadows_checkbox.grid_remove()

        # NDWI Threshold Input (Initially Hidden)
        self.post_ndwi_label = ttk.Label(self.post_frame, text="NDWI Threshold:", width=label_width, anchor="e")
        self.post_ndwi_entry = ttk.Entry(self.post_frame, width=7)
        self.post_ndwi_entry.insert(0, "0.01")  # Default value
        self.post_ndwi_label.grid(row=10, column=1, padx=0, pady=5, sticky="e")
        self.post_ndwi_entry.grid(row=10, column=2, padx=0, pady=5, sticky="w")
        self.post_ndwi_label.grid_remove()
        self.post_ndwi_entry.grid_remove()

        # Cloud Masking Method (Initially Hidden)
        self.post_cloud_method_label = ttk.Label(self.post_frame, text="Cloud Masking Method:", width=label_width, anchor="w")
        self.post_cloud_method_var = tk.StringVar(value="standard")
        self.post_cloud_method_dropdown = ttk.Combobox(self.post_frame, textvariable=self.post_cloud_method_var, state="readonly", width=15)
        self.post_cloud_method_dropdown.grid(row=11, column=1, padx=5, pady=5, sticky="w")
        self.post_cloud_method_label.grid(row=11, column=0, padx=5, pady=5, sticky="w")
        self.post_cloud_method_dropdown.bind("<<ComboboxSelected>>", self.post_update_warning)
        self.post_cloud_method_dropdown.grid_remove()
        self.post_cloud_method_label.grid_remove()

        # Warning Label (Initially Hidden)
        self.post_warning_label = ttk.Label(self.post_frame, text="This method may take a long time!", foreground="red", )
        self.post_warning_label.grid(row=11, column=1, padx=0, pady=5, sticky="e")
        self.post_warning_label.grid_remove()  # Hide it by default

        self.post_update_cloud_methods()

        # Save Preprocessed Image Path (Initially Hidden)
        self.post_preprocess_label = ttk.Label(self.post_frame, text="Save Prep. Image As:", width=label_width, anchor="w")
        self.post_preprocess_entry = ttk.Entry(self.post_frame, width=50)
        self.post_preprocess_button = ttk.Button(self.post_frame, text="...", command=self.post_select_preprocessed_path, width=3)

        self.post_preprocess_label.grid(row=12, column=0, padx=5, pady=5, sticky="w")
        self.post_preprocess_entry.grid(row=12, column=1, padx=5, pady=5)
        self.post_preprocess_button.grid(row=12, column=2, padx=5, pady=5)
        self.post_preprocess_label.grid_remove()
        self.post_preprocess_entry.grid_remove()
        self.post_preprocess_button.grid_remove()
        
        # Model Options
        self.model_frame = ttk.LabelFrame(root, text="Model Settings", padding=(10, 5, 10, 5))
        self.model_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Load Model
        ttk.Label(self.model_frame, text="Select Model File (.pkl):", width=label_width, anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_entry = ttk.Entry(self.model_frame, width=50)
        self.model_entry.grid(row=0, column=1, padx=5, pady=5)
        self.model_button = ttk.Button(self.model_frame, text="...", command=self.load_model_path, width=3)
        self.model_button.grid(row=0, column=2, padx=5, pady=5)

        # Number of Samples
        ttk.Label(self.model_frame, text="Number of Samples:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.samples_entry = ttk.Entry(self.model_frame, width=10)
        self.samples_entry.insert(0, "1000")
        self.samples_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Threshold dNBR (Defaults to 300)
        ttk.Label(self.model_frame, text="Threshold dNBR:").grid(row=1, column=1, padx=5, pady=5, sticky="e")
        self.threshold_entry = ttk.Entry(self.model_frame, width=4)
        self.threshold_entry.insert(0, "300")  # Default value
        self.threshold_entry.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Classify frame
        self.classify_frame = ttk.LabelFrame(root, text="Classification Settings", padding=(10, 5, 10, 5))
        self.classify_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Classify checkbox
        self.classify_var = tk.BooleanVar(value=False)
        self.classify_after_check = ttk.Checkbutton(self.classify_frame, text="Classify after image", variable=self.classify_var, command=self.toggle_classify_output)
        self.classify_after_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Display dNBR checkbox
        self.display_dnbr_var = tk.BooleanVar(value=False)  # Default: Show dNBR
        self.display_dnbr_checkbox = ttk.Checkbutton(self.classify_frame, text="Display dNBR", variable=self.display_dnbr_var)
        self.display_dnbr_checkbox.grid(row=0, column=1)

        # Classified Image Output Path (Hidden Initially)
        self.classify_label = ttk.Label(self.classify_frame, text="Save Classified Image as:", width=label_width, anchor="w")
        self.classify_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.output_entry = ttk.Entry(self.classify_frame, width=50)
        self.output_entry.grid(row=1, column=1, padx=5, pady=5)
        self.output_button = ttk.Button(self.classify_frame, text="...", command=self.select_output_path, width=3)
        self.output_button.grid(row=1, column=2, padx=5, pady=5)
        self.classify_label.grid_remove()
        self.output_button.grid_remove()
        self.output_entry.grid_remove()
        
        # Running Frame
        self.running_frame = ttk.Frame(root, padding=(10, 5, 10, 5))
        self.running_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Run Button
        self.classify_button = ttk.Button(self.running_frame, text="Test Model", command=self.run_testing_and_classification, width=20)
        self.classify_button.grid(row=0, column=0, pady=0, padx=200)
        
        #Status Label
        self.status_label = ttk.Label(self.running_frame, text="", foreground="blue")  # Status message
        self.status_label.grid(row=1, column=0, pady=5, padx=200)  # Place under the classify button

        # Go Back Button 
        self.back_button = ttk.Button(root, text="<-", command=self.go_back, width=5)
        self.back_button.grid(row=7, column=0, padx=10, pady=1, sticky="sw")  # Bottom left placement
        

    def toggle_classify_output(self):
        """Show or hide the classified image output path widgets
        based on the state of classify_after_var."""
        if self.classify_var.get():
            self.classify_label.grid()
            self.output_entry.grid()
            self.output_button.grid()
        else:
            self.classify_label.grid_remove()
            self.output_entry.grid_remove()
            self.output_button.grid_remove()
    

    def pre_toggle_preprocessing_path(self):
        if self.pre_mask_clouds_var.get():
            self.pre_cloud_shadows_checkbox.grid()
            self.pre_cloud_method_dropdown.grid()
            self.pre_cloud_method_label.grid()
        else:
            self.pre_cloud_shadows_checkbox.grid_remove()
            self.pre_cloud_method_dropdown.grid_remove()
            self.pre_cloud_method_label.grid_remove()

        if self.pre_mask_water_var.get():
            self.pre_ndwi_label.grid()
            self.pre_ndwi_entry.grid()
        else:
            self.pre_ndwi_label.grid_remove()
            self.pre_ndwi_entry.grid_remove()

        if self.pre_mask_clouds_var.get() or self.pre_mask_water_var.get():
            self.pre_preprocess_label.grid()
            self.pre_preprocess_entry.grid()
            self.pre_preprocess_button.grid()
        else:
            self.pre_preprocess_label.grid_remove()
            self.pre_preprocess_entry.grid_remove()
            self.pre_preprocess_button.grid_remove()


    def load_pre_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Raster Files", "*.tif;*.tiff;*.img;*.jp2")])
        if file_path:
            self.pre_entry.delete(0, tk.END)
            self.pre_entry.insert(0, file_path)
            self.pre_nodata_entry.delete(0, tk.END)
            detected_nodata = preprocessing.detect_nodata_from_path(file_path)
            if detected_nodata is not None:
                self.pre_nodata_entry.delete(0, tk.END)
                self.pre_nodata_entry.insert(0, str(detected_nodata))
    
    def load_pre_folder(self):
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if not folder_path:
            return
        
        satellite = self.satellite_var.get()
        self.update_state_label("Generating Stacked Raster for Pre Image...")
        
        try:
            stacked_image_path = image_loader.load_image(folder_path, satellite)
            self.update_state_label("Stacked Raster for Pre Image Generated!")
            self.pre_entry.delete(0, tk.END)
            self.pre_entry.insert(0, stacked_image_path)
            # Detect NoData value if applicable
            detected_nodata = preprocessing.detect_nodata_from_path(stacked_image_path)
            self.pre_nodata_entry.delete(0, tk.END)
            if detected_nodata is not None:
                self.pre_nodata_entry.delete(0, tk.END)
                self.pre_nodata_entry.insert(0, str(detected_nodata))

        except Exception as e:
            self.update_state_label("Error generating stacked raster for Pre Image")
            messagebox.showerror("Error", f"Failed to stack raster for Pre Image: {e}")

    def pre_preview(self):
        image_path = self.pre_entry.get().strip()
        nodata_value = self.pre_nodata_entry.get().strip()
        if not image_path:
            messagebox.showerror("Error", "Please select an image file.")
            return
        # Call the function to display the preview
        self.display_rgb_image(image_path, nodata_value if nodata_value else None)


    def pre_select_preprocessed_path(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF Files", "*.tif")])
        if file_path:
            self.pre_preprocess_entry.delete(0, tk.END)
            self.pre_preprocess_entry.insert(0, file_path)

    def update_cloud_methods(self, event=None):
        self.pre_update_cloud_methods()
        self.post_update_cloud_methods()

    def pre_update_cloud_methods(self, event=None):
        selected_satellite = self.satellite_var.get()
        mask_shadows = self.pre_mask_cloud_shadows_var.get()
        
        if selected_satellite == "Sentinel-2":
            methods = ["standard", "auto", "scl", "omnicloudmask"] if mask_shadows else ["standard", "auto", "probability", "scl", "qa", "omnicloudmask"]
        elif selected_satellite == "Landsat 8/9":
            methods = ["qa", "auto", "omnicloudmask"]
        else:
            methods = []
        
        self.pre_cloud_method_dropdown["values"] = methods
        self.pre_cloud_method_var.set(methods[0])
        self.pre_update_warning()

    def pre_update_warning(self, event=None):
        if self.pre_cloud_method_var.get() in ["omnicloudmask", "auto"]:
            self.pre_warning_label.grid()
        else:
            self.pre_warning_label.grid_remove()

    def post_toggle_preprocessing_path(self):
        if self.post_mask_clouds_var.get():
            self.post_cloud_shadows_checkbox.grid()
            self.post_cloud_method_dropdown.grid()
            self.post_cloud_method_label.grid()
        else:
            self.post_cloud_shadows_checkbox.grid_remove()
            self.post_cloud_method_dropdown.grid_remove()
            self.post_cloud_method_label.grid_remove()

        if self.post_mask_water_var.get():
            self.post_ndwi_label.grid()
            self.post_ndwi_entry.grid()
        else:
            self.post_ndwi_label.grid_remove()
            self.post_ndwi_entry.grid_remove()

        if self.post_mask_clouds_var.get() or self.post_mask_water_var.get():
            self.post_preprocess_label.grid()
            self.post_preprocess_entry.grid()
            self.post_preprocess_button.grid()
        else:
            self.post_preprocess_label.grid_remove()
            self.post_preprocess_entry.grid_remove()
            self.post_preprocess_button.grid_remove()

    def load_post_folder(self):
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if not folder_path:
            return
        
        satellite = self.satellite_var.get()
        self.update_state_label("Generating Stacked Raster for Post Image...")
        
        try:
            stacked_image_path = image_loader.load_image(folder_path, satellite)
            self.update_state_label("Stacked Raster for Post Image Generated!")
            self.post_entry.delete(0, tk.END)
            self.post_entry.insert(0, stacked_image_path)
            # Detect NoData value if applicable
            detected_nodata = preprocessing.detect_nodata_from_path(stacked_image_path)
            self.post_nodata_entry.delete(0, tk.END)
            if detected_nodata is not None:
                self.post_nodata_entry.delete(0, tk.END)
                self.post_nodata_entry.insert(0, str(detected_nodata))

        except Exception as e:
            self.update_state_label("Error generating stacked raster for Post Image")
            messagebox.showerror("Error", f"Failed to stack raster for Post Image: {e}")

    def load_post_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Raster Files", "*.tif;*.tiff;*.img;*.jp2")])
        if file_path:
            self.post_entry.delete(0, tk.END)
            self.post_entry.insert(0, file_path)
            self.post_nodata_entry.delete(0, tk.END)
            detected_nodata = preprocessing.detect_nodata_from_path(file_path)
            if detected_nodata is not None:
                self.post_nodata_entry.delete(0, tk.END)
                self.post_nodata_entry.insert(0, str(detected_nodata))

    def post_preview(self):
        image_path = self.post_entry.get().strip()
        nodata_value = self.post_nodata_entry.get().strip()
        if not image_path:
            messagebox.showerror("Error", "Please select an image file.")
            return
        # Call the function to display the preview
        self.display_rgb_image(image_path, nodata_value if nodata_value else None)

    def post_select_preprocessed_path(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF Files", "*.tif")])
        if file_path:
            self.post_preprocess_entry.delete(0, tk.END)
            self.post_preprocess_entry.insert(0, file_path)

    def post_update_cloud_methods(self, event=None):
        selected_satellite = self.satellite_var.get()
        mask_shadows = self.post_mask_cloud_shadows_var.get()
        
        if selected_satellite == "Sentinel-2":
            methods = ["standard", "auto", "scl", "omnicloudmask"] if mask_shadows else ["standard", "auto", "probability", "scl", "qa", "omnicloudmask"]
        elif selected_satellite == "Landsat 8/9":
            methods = ["qa", "auto", "omnicloudmask"]
        else:
            methods = []
        
        self.post_cloud_method_dropdown["values"] = methods
        self.post_cloud_method_var.set(methods[0])
        self.post_update_warning()

    def post_update_warning(self, event=None):
        if self.post_cloud_method_var.get() in ["omnicloudmask", "auto"]:
            self.post_warning_label.grid()
        else:
            self.post_warning_label.grid_remove()

    def load_model_path(self):
        """Open file dialog to select a model file"""
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, file_path)

    def load_model(self):
        """Loads the classification model"""        
        try:
            self.model = processing.load_model(self.model_path)
        except Exception as e:
            self.update_state_label("Error loading model")
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        # Proceed to Makings
        self.update_state_label("Applying Masks...")
        self.root.after(100, self.apply_masks)  # Move to next step asynchronously

    def select_output_path(self):
        """Open file dialog to specify where to save the classified image"""
        file_path = filedialog.asksaveasfilename(defaultextension=".tif",
                                                 filetypes=[("TIFF Files", "*.tif")])
        if file_path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, file_path)

    def go_back(self):
        """Destroy the current UI and go back to the main menu"""
        for widget in self.root.winfo_children():
            widget.destroy()  # Clear the window
        BurnAreaClassifierApp(self.root)  # Reinitialize the main menu

    def update_state_label(self, content=""):
        self.status_label.config(text=content)
        self.root.update_idletasks()  # Refresh UI immediately

    def run_testing_and_classification(self):
        """Starts the full training and classification process."""

        # Get input parameters from UI
        self.model_path = self.model_entry.get()
        self.pre_image_path = self.pre_entry.get()
        self.post_image_path = self.post_entry.get()
        try:
            self.pre_nodata = float(self.pre_nodata_entry.get()) if self.pre_nodata_entry.get().strip() else None
        except ValueError:
            self.pre_nodata = None  # Defaults to None if input is invalid
        try:
            self.post_nodata = float(self.post_nodata_entry.get()) if self.post_nodata_entry.get().strip() else None
        except ValueError:
            self.post_nodata = None  # Defaults to None if input is invalid
        self.samples = int(self.samples_entry.get()) if self.samples_entry.get().isdigit() else 1000

        self.pre_preprocess_path = self.pre_preprocess_entry.get() if self.pre_preprocess_entry.winfo_ismapped() else None
        self.post_preprocess_path = self.post_preprocess_entry.get() if self.post_preprocess_entry.winfo_ismapped() else None

        self.threshold = int(self.threshold_entry.get()) if self.threshold_entry.get().isdigit() else 300

        satellite_mapping = {
            "Sentinel-2": "S2",
            "Landsat 8/9": "L8"
        }

        # Get selected value from dropdown
        self.selected_satellite = self.satellite_var.get()
        self.satellite = satellite_mapping.get(self.selected_satellite, "S2")  # Default to "S2" if not found
        self.output_path = self.output_entry.get()

        if not self.pre_image_path or not self.post_image_path:
            messagebox.showerror("Error", "Please select both Before and After images.")
            return
        if not self.model_path:
            messagebox.showerror("Error", "Please select a model to test.")
            return
        

        self.update_state_label("Loading Model...")
        self.root.after(100, self.load_model)


    def apply_masks(self):
    
        # Create and start threads for preprocessing
        self.pre_thread = threading.Thread(target=self.preprocess_image, args=(self.pre_image_path, self.pre_nodata, "pre"))
        self.post_thread = threading.Thread(target=self.preprocess_image, args=(self.post_image_path, self.post_nodata, "post"))

        self.pre_thread.start()
        self.post_thread.start()

        # Schedule a check to see if both threads are done
        self.root.after(100, self.check_preprocessing_done)

    def preprocess_image(self, image_path, nodata, image_type):
        """Preprocesses the Before and After images."""

        apply_mask = (
            self.pre_mask_clouds_var.get() if image_type == "pre" else self.post_mask_clouds_var.get()
        ) or (
            self.pre_mask_water_var.get() if image_type == "pre" else self.post_mask_water_var.get()
        )

        if apply_mask:        
            try:

                preprocess_path = self.pre_preprocess_path if image_type == "pre" else self.post_preprocess_path

                processed_path = preprocessing.apply_masks(
                    image_path, self.satellite, 
                    preprocess_path if preprocess_path else None,
                    self.pre_cloud_method_var.get() if image_type == "pre" else self.post_cloud_method_var.get(),
                    self.pre_mask_clouds_var.get() if image_type == "pre" else self.post_mask_clouds_var.get(),
                    self.pre_mask_cloud_shadows_var.get() if image_type == "pre" else self.post_mask_cloud_shadows_var.get(),
                    self.pre_mask_water_var.get() if image_type == "pre" else self.post_mask_water_var.get(),
                    self.pre_ndwi_entry.get() if image_type == "pre" else self.post_ndwi_entry.get(),
                    nodata
                )

                if image_type == "pre":
                    self.pre_image_path = processed_path
                else:
                    self.post_image_path = processed_path

            except Exception as e:
                messagebox.showerror(f"Preprocessing Error - {image_type.capitalize()} Image", f"Failed to preprocess {image_type} image: {e}")

    def check_preprocessing_done(self):
        """Checks if preprocessing threads have finished and moves to the next step."""
        if self.pre_thread.is_alive() or self.post_thread.is_alive():
            # If either thread is still running, check again in 100ms
            self.root.after(100, self.check_preprocessing_done)
        else:
            # Both threads have finished, move to the next step
            if not self.pre_image_path:
                    self.update_state_label("Masking Error on Pre Fire Image!")
                    messagebox.showerror("Masking Error", "An error happened while applying the masks to Pre Image. \nPlease make sure the needed bands for the selected method is present")
                    return
            if not self.post_image_path:
                    self.update_state_label("Masking Error on Post Fire Image!")
                    messagebox.showerror("Masking Error", "An error happened while applying the masks to Post Image. \nPlease make sure the needed bands for the selected method is present")
                    return
            self.update_state_label("Computing NBR and dNBR...")
            self.root.after(100, self.compute_nbr_and_dNBR)

    def compute_nbr_and_dNBR(self):
        """Computes NBR and dNBR, then moves to testing."""
        try:
            pre_NBR = preprocessing.compute_NBR(self.pre_image_path, self.satellite, self.pre_nodata)
            post_NBR = preprocessing.compute_NBR(self.post_image_path, self.satellite, self.post_nodata)
            self.dNBR = processing.compute_dNBR(pre_NBR, post_NBR)

            if self.display_dnbr_var.get():
                self.update_state_label("Displaying dNBR...")
                self.root.after(100, self.display_dNBR)

            else:
                self.update_state_label("Creating Testing Dataset...")
                self.root.after(100, self.create_testing_set)
    
        except Exception as e:
            self.update_state_label("Error computing NBR")
            messagebox.showerror("Error", f"Failed to compute NBR: {e}")

    def display_dNBR(self):
        try:
            img_height, img_width = self.dNBR.shape
            aspect_ratio = img_width / img_height
            base_size = 6  # Base size for the figure
            fig = plt.figure(figsize=(base_size * aspect_ratio, base_size) if aspect_ratio > 1 else (base_size, base_size / aspect_ratio))

            cmap = mcolors.LinearSegmentedColormap.from_list("dNBR", ['darkgreen', 'green', 'yellow', 'orange', 'red', 'purple'])
            norm = mcolors.Normalize(vmin=-500, vmax=1300)
            plt.gca().set_aspect(self.dNBR.shape[1] / self.dNBR.shape[0])
            im = plt.imshow(self.dNBR, cmap=cmap, norm=norm) 
            cbar = plt.colorbar(im, orientation="horizontal", fraction=0.03, pad=0.1)
            cbar.set_label("dNBR Value")
            fig.canvas.manager.set_window_title("Classified Image")  # Set window title
            plt.title("Raw dNBR Output")
            plt.show()

        except Exception as e:
            self.update_state_label("Error displaying dNBR")
            messagebox.showerror("Error", f"Failed to display dNBR: {e}")
            
        finally:
            self.update_state_label("Creating Testing Dataset...")
            self.root.after(100, self.create_testing_set)

    def create_testing_set(self):
        """Creates testing data and moves to model testing."""
        try:
            self.X_sampled, self.y_sampled = processing.create_training_set(
                self.post_image_path, self.dNBR, method='threshold', threshold=self.threshold,
                samples=self.samples, post_image_nodata=self.post_nodata
            )

            self.update_state_label("Testing Model...")
            self.root.after(100, self.test_model)

        except Exception as e:
            self.update_state_label("Error creating the testing set!")
            messagebox.showerror("Error", f"Failed to create testing set: {e}")

    def test_model(self):
        
        try:
            self.conf_matrix, self.class_report, self.accuracy = processing.test_model(
                self.model, self.X_sampled, self.y_sampled
            )

            self.update_state_label("Testing complete! Displaying metrics..")
            self.root.after(100, self.display_metrics)


        except Exception as e:
            self.update_state_label("Error testing the model!")
            messagebox.showerror("Error", f"Failed to test the model: {e}")


    def display_metrics(self):
        """Displays training metrics in a structured, modern format using ttk."""
        
        # Check if class "1.0" exists, otherwise default to 0
        precision_0 = self.class_report.get("0.0", {}).get("precision", 0)
        recall_0 = self.class_report.get("0.0", {}).get("recall", 0)
        f1_score_0 = self.class_report.get("0.0", {}).get("f1-score", 0)
        support_0 = self.class_report.get("0.0", {}).get("support", 0)

        precision_1 = self.class_report.get("1.0", {}).get("precision", 0)
        recall_1 = self.class_report.get("1.0", {}).get("recall", 0)
        f1_score_1 = self.class_report.get("1.0", {}).get("f1-score", 0)
        support_1 = self.class_report.get("1.0", {}).get("support", 0)

        accuracy = self.class_report.get("accuracy", 0)

        # Create a new Toplevel window
        metrics_window = Toplevel(self.root)
        metrics_window.title("Testing Results")
        metrics_window.geometry("500x350")  # Adjusted window size

        # Create a frame for layout using ttk
        frame = ttk.Frame(metrics_window, padding=10)
        frame.pack(fill="both", expand=True)

        # Title Label
        ttk.Label(frame, text="Model Testing Report", font=("Arial", 14, "bold")).pack(pady=5)
        ttk.Label(frame, text=f"Overall Accuracy: {accuracy:.4f}", font=("Arial", 12, "bold")).pack()

        # Table for Classification Report
        table_frame = ttk.Frame(frame)
        table_frame.pack(pady=5)

        columns = ("Class", "Precision", "Recall", "F1-Score", "Support")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=3)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=80)

        # Insert rows for each class
        tree.insert("", "end", values=("Unburned (0)", f"{precision_0:.4f}", f"{recall_0:.4f}", f"{f1_score_0:.4f}", int(support_0)))
        tree.insert("", "end", values=("Burned (1)", f"{precision_1:.4f}", f"{recall_1:.4f}", f"{f1_score_1:.4f}", int(support_1)))

        tree.pack()

        # Confusion Matrix Section
        ttk.Label(frame, text="Confusion Matrix:", font=("Arial", 12, "bold")).pack()

        matrix_frame = ttk.Frame(frame, padding=5)
        matrix_frame.pack()

        # Confusion Matrix Table Headers
        labels = ["Actual 0", "Actual 1"]
        headers = ["Predicted 0", "Predicted 1"]

        for i, header in enumerate([""] + headers):
            ttk.Label(matrix_frame, text=header, font=("Arial", 10, "bold"), borderwidth=2, relief="ridge", width=12, anchor="center").grid(row=0, column=i, sticky="nsew", padx=2, pady=2)

        # Confusion Matrix Rows
        for i, label in enumerate(labels):
            ttk.Label(matrix_frame, text=label, font=("Arial", 10, "bold"), borderwidth=2, relief="ridge", width=12, anchor="center").grid(row=i+1, column=0, sticky="nsew", padx=2, pady=2)
            for j in range(2):
                ttk.Label(matrix_frame, text=str(self.conf_matrix[i][j]), font=("Arial", 10), borderwidth=2, relief="ridge", width=12, anchor="center").grid(row=i+1, column=j+1, sticky="nsew", padx=2, pady=2)

        # Close Button
        ttk.Button(frame, text="Close", command=metrics_window.destroy).pack(pady=10)

        if self.classify_var.get():
            self.update_state_label("Classifying Image...")
            self.root.after(100, self.classify_image)

    def classify_image(self):
        """Runs classification using the trained model."""
        try:
            self.classified_image_path = processing.classify(
                self.model, self.post_image_path, output_path = self.output_path if self.output_path else None,
                nodata_value=float(self.post_nodata) if self.post_nodata else None
            )

            self.update_state_label("Classification complete")

            # Ask if the user wants to preview
            open_preview = messagebox.askyesno(
                "Classification Complete",
                f"Classified image saved at {self.classified_image_path}\n\nDo you want to preview the image?"
            )

            if open_preview:
                self.root.after(100, lambda: self.display_classified_image(self.classified_image_path))

        except Exception as e:
            self.update_state_label("Classification failed")
            messagebox.showerror("Error", f"Classification failed: {e}")

    def display_classified_image(self, image_path):
        """Displays the classified image with a custom title and legend."""
        
        

        with rio.open(image_path) as src:
            classified_data = src.read(1)  # Read the first band

            img_height, img_width = classified_data.shape  # Get image size
            aspect_ratio = img_width / img_height
            base_size = 6  # Base figure size

            fig = plt.figure(figsize=(base_size * aspect_ratio, base_size) if aspect_ratio > 1 else (base_size, base_size / aspect_ratio))  # Create a figure
            fig.canvas.manager.set_window_title("Classified Image")  # Set window title
            

            ax = fig.add_subplot(111)  # Add subplot
            cmap = mcolors.ListedColormap(["green", "red"])  # 0 = Green (Unburned), 1 = Red (Burned)
            bounds = [0, 0.5, 1]  # Define boundaries for values
            norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Normalize colors to values

            img = ax.imshow(classified_data, cmap=cmap, norm=norm)
            ax.set_title("Classified Image")

            # Add legend
            legend_labels = [
                mpatches.Patch(color="green", label="Unburned (0)"),
                mpatches.Patch(color="red", label="Burned (1)")
            ]
            ax.legend(handles=legend_labels, loc="upper right")

        plt.show()  # Display the figure

    def display_rgb_image(self, image_path, nodata_value):
        """Displays an RGB preview using band descriptions to match Red, Green, and Blue bands."""
        try:
            with rio.open(image_path) as src:
                band_descriptions = src.descriptions  # Get band names

                if not band_descriptions or all(desc is None for desc in band_descriptions):
                    messagebox.showerror("Error", "No band descriptions found. Cannot determine RGB bands.")
                    return

                # Define band name variations for different satellites
                if self.satellite_var.get() == "Sentinel-2":
                    red_band_names = ["B4", "B04"]
                    green_band_names = ["B3", "B03"]
                    blue_band_names = ["B2", "B02"]
                elif self.satellite_var.get() == "Landsat 8/9":
                    red_band_names = ["B4", "SR_B4"]
                    green_band_names = ["B3", "SR_B3"]
                    blue_band_names = ["B2", "SR_B2"]
                else:
                    messagebox.showerror("Error", "Unknown satellite. Cannot determine RGB bands.")
                    return

                # Find the correct band indices dynamically
                band_indices = {"Red": None, "Green": None, "Blue": None}

                for i, desc in enumerate(band_descriptions):
                    if any(red in desc for red in red_band_names):
                        band_indices["Red"] = i + 1  # Rasterio uses 1-based indexing
                    elif any(green in desc for green in green_band_names):
                        band_indices["Green"] = i + 1
                    elif any(blue in desc for blue in blue_band_names):
                        band_indices["Blue"] = i + 1
                # Ensure all bands are found
                if None in band_indices.values():
                    messagebox.showerror("Error", "Could not find all RGB bands in the metadata.")
                    return

                # Read the bands
                r, g, b = band_indices["Red"], band_indices["Green"], band_indices["Blue"]
                red, green, blue = src.read(r).astype(np.float32), src.read(g).astype(np.float32), src.read(b).astype(np.float32)

                if nodata_value is not None and np.isnan(float(nodata_value)):
                    mask = np.isnan(red) | np.isnan(green) | np.isnan(blue)
                elif nodata_value is not None:
                    mask = (red == nodata_value) | (green == nodata_value) | (blue == nodata_value)
                else:
                    mask = np.zeros_like(red, dtype=bool)  # No NoData value in metadata, assume all valid
            
                # Normalize bands
                def normalize_band(band, mask):
                    band_min, band_max = np.min(band[~mask]), np.max(band[~mask])
                    normalized_band = (band - band_min) / (band_max - band_min)
                    normalized_band[mask] = 0  # Keep NoData areas black
                    return normalized_band

                red, green, blue = normalize_band(red, mask), normalize_band(green, mask), normalize_band(blue, mask)

                def gamma_correction(band, gamma=1.2):  # Default gamma 1.2 makes midtones brighter
                    return np.power(band, 1/gamma)

                red, green, blue = gamma_correction(red), gamma_correction(green), gamma_correction(blue)

                img_height, img_width = red.shape  # Get image size
                aspect_ratio = img_width / img_height
                base_size = 6  # Base figure size

                # Stack RGB and display
                rgb_image = np.dstack((red, green, blue))
                rgb_image[mask] = [1, 1, 1]  # White color for NoData area
                fig, ax = plt.subplots(figsize=(base_size * aspect_ratio, base_size) if aspect_ratio > 1 else (base_size, base_size / aspect_ratio))
                fig.canvas.manager.set_window_title("Image Preview")
                ax.imshow(rgb_image)
                ax.set_title("RGB Composite Preview")
                plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Could not display RGB preview: {e}")


# Start Application
if __name__ == "__main__":
    root = tk.Tk()
    app = BurnAreaClassifierApp(root)
    root.mainloop()
