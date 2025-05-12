import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from typing import Callable, Dict, List, Tuple, Union, Optional
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.effects = {
            "Color Effects": {
                "Sepia": self.effect_sepia,
                "Negative": self.effect_negative,
                "Sobel": self.effect_sobel,
            },
            "Artistic Effects": {
                "Box Blur": self.effect_box_blur,
                "Sharpening": self.effect_sharpening,
                "Edge Sketch": self.effect_edge_sketch,
            },
            "Special Effects": {
                "HDR": self.effect_hdr,
                "Vignette": self.effect_vignette,
                "Laplacian": self.effect_laplacian,
            }
        }

    def load_image(self, file_path: str) -> None:
        """Load an image from file path."""
        self.original_image = cv2.imread(file_path)
        self.current_image = self.original_image.copy()
        return self.original_image

    def save_image(self, file_path: str) -> None:
        """Save the current image to file path."""
        if self.current_image is not None:
            cv2.imwrite(file_path, self.current_image)

    def apply_effect(self, effect_func: Callable, intensity: float = 1.0) -> None:
        """Apply an effect to the current image."""
        if self.original_image is not None:
            self.current_image = effect_func(self.original_image, intensity)

    def reset_image(self) -> None:
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            
    def resize_image(self, width: int, height: int) -> None:
        if self.current_image is not None:
            # Convert OpenCV image to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
            # Resize the image
            resized_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            # Convert back to OpenCV format
            self.current_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
            
    def flip_image(self, direction: str) -> None:
        if self.current_image is not None:
            # Convert OpenCV image to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
            
            # Flip the image based on direction
            if direction.lower() == 'horizontal':
                flipped_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            elif direction.lower() == 'vertical':
                flipped_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                raise ValueError("Direction must be 'horizontal' or 'vertical'")
                
            # Convert back to OpenCV format
            self.current_image = cv2.cvtColor(np.array(flipped_image), cv2.COLOR_RGB2BGR)

    # Effect implementations
    def effect_sepia(self, img: np.ndarray, intensity: float) -> np.ndarray:
        img_float = np.array(img, dtype=np.float64)
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                               [0.349, 0.686, 0.168],
                               [0.272, 0.534, 0.131]])
        sepia_img = cv2.transform(img_float, sepia_filter)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return cv2.addWeighted(sepia_img, intensity, img, 1 - intensity, 0)

    def effect_negative(self, img: np.ndarray, intensity: float) -> np.ndarray:
        # Invert the image
        negative = 255 - img
        
        # Blend with original based on intensity
        return cv2.addWeighted(negative, intensity, img, 1 - intensity, 0)

    def effect_box_blur(self, img: np.ndarray, intensity: float) -> np.ndarray:
        # Calculate kernel size based on intensity (3 to 15)
        kernel_size = int(3 + intensity * 12)
        # Ensure kernel size is odd
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Apply box blur
        blurred = cv2.blur(img, (kernel_size, kernel_size))
        
        # Blend with original based on intensity
        return cv2.addWeighted(blurred, intensity, img, 1 - intensity, 0)

    def effect_sharpening(self, img: np.ndarray, intensity: float) -> np.ndarray:
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        # Apply sharpening
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # Blend with original based on intensity
        return cv2.addWeighted(sharpened, intensity, img, 1 - intensity, 0)

    def effect_edge_sketch(self, img: np.ndarray, intensity: float) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_inv = cv2.bitwise_not(edges)
        edges_inv_color = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(edges_inv_color, intensity, img, 1 - intensity, 0)

    def effect_hdr(self, img: np.ndarray, intensity: float) -> np.ndarray:
        hdr = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        return cv2.addWeighted(hdr, intensity, img, 1 - intensity, 0)

    def effect_sobel(self, img: np.ndarray, intensity: float) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Sobel operators
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-255
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = magnitude.astype(np.uint8)
        
        # Convert back to BGR
        sobel_edges = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
        
        # Blend with original based on intensity
        return cv2.addWeighted(sobel_edges, intensity, img, 1 - intensity, 0)

    def effect_vignette(self, img: np.ndarray, intensity: float) -> np.ndarray:
        return self.apply_vignette(img, intensity * 0.5)

    def apply_vignette(self, img: np.ndarray, level: float) -> np.ndarray:
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols * level)
        kernel_y = cv2.getGaussianKernel(rows, rows * level)
        mask = kernel_y * kernel_x.T
        mask = mask / mask.max()
        vignette = np.empty_like(img)
        for i in range(3):
            vignette[:,:,i] = img[:,:,i] * mask
        return vignette

    def apply_adjustments(self, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
        if self.current_image is None:
            return None
            
        # Create a copy of the current image
        adjusted = self.current_image.copy()
        
        # Apply brightness adjustment
        if brightness != 0:
            adjusted = cv2.add(adjusted, brightness)
            
        # Apply contrast adjustment
        if contrast != 1.0:
            adjusted = cv2.multiply(adjusted, contrast)
            
        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted

    def apply_adjustments_to_image(self, image: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
        if image is None:
            return None

        adjusted = image.copy().astype(np.float32)
        if brightness != 0:
            adjusted += brightness
        if contrast != 1.0:
            adjusted *= contrast
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted


    def calculate_histograms(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.current_image is None:
            return None, None, None
            
        # Calculate histograms for each channel
        hist_b = cv2.calcHist([self.current_image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([self.current_image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([self.current_image], [2], None, [256], [0, 256])
        
        # Normalize histograms
        hist_b = cv2.normalize(hist_b, None, 0, 1, cv2.NORM_MINMAX)
        hist_g = cv2.normalize(hist_g, None, 0, 1, cv2.NORM_MINMAX)
        hist_r = cv2.normalize(hist_r, None, 0, 1, cv2.NORM_MINMAX)
        
        return hist_b, hist_g, hist_r

    def effect_laplacian(self, img: np.ndarray, intensity: float) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        laplacian_color = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(laplacian_color, intensity, img, 1 - intensity, 0)

class ImageEditorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Image Editor")
        self.geometry("1600x900")
        self.minsize(1200, 800)

        # Initialize image processor
        self.image_processor = ImageProcessor()
        
        # Initialize adjustment variables
        self.brightness_var = ctk.DoubleVar(value=0.0)
        self.contrast_var = ctk.DoubleVar(value=1.0)
        
        # Create main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Create left panel for image display
        self.image_frame = ctk.CTkFrame(self.main_container)
        self.image_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="No image loaded")
        self.image_label.pack(expand=True)

        # Create right panel for controls and histogram
        self.right_panel = ctk.CTkFrame(self.main_container)
        self.right_panel.pack(side="right", fill="y", padx=(10, 0))

        # Create histogram frame
        self.histogram_frame = ctk.CTkFrame(self.right_panel)
        self.histogram_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Create matplotlib figure for histogram
        self.hist_figure = Figure(figsize=(4, 2), dpi=100)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, master=self.histogram_frame)
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Create control frame
        self.control_frame = ctk.CTkFrame(self.right_panel)
        self.control_frame.pack(fill="y", padx=10, pady=5)

        # Create buttons frame
        self.buttons_frame = ctk.CTkFrame(self.control_frame)
        self.buttons_frame.pack(fill="x", padx=10, pady=(10, 5))

        # Add file operation buttons
        self.load_button = ctk.CTkButton(
            self.buttons_frame,
            text="Load Image",
            command=self.load_image
        )
        self.load_button.pack(fill="x", pady=(0, 5))

        self.save_button = ctk.CTkButton(
            self.buttons_frame,
            text="Save Image",
            command=self.save_image,
            state="disabled",
            fg_color="#28a745",  # Green color
            hover_color="#218838"  # Darker green for hover
        )
        self.save_button.pack(fill="x", pady=(0, 5))

        # Add format selection combobox
        self.format_frame = ctk.CTkFrame(self.buttons_frame)
        self.format_frame.pack(fill="x", pady=(0, 5))
        
        self.format_label = ctk.CTkLabel(
            self.format_frame,
            text="Save Format:"
        )
        self.format_label.pack(side="left", padx=(0, 5))
        
        self.format_var = ctk.StringVar(value="PNG")
        self.format_combobox = ctk.CTkComboBox(
            self.format_frame,
            values=["PNG", "JPEG", "BMP"],
            variable=self.format_var,
            state="disabled"
        )
        self.format_combobox.pack(side="left", fill="x", expand=True)

        self.reset_button = ctk.CTkButton(
            self.buttons_frame,
            text="Reset Image",
            command=self.reset_image,
            state="disabled",
            fg_color="#dc3545",  # Red color
            hover_color="#c82333"  # Darker red for hover
        )
        self.reset_button.pack(fill="x")

        # Create adjustments frame
        self.adjustments_frame = ctk.CTkFrame(self.control_frame)
        self.adjustments_frame.pack(fill="x", padx=10, pady=5)
        
        # Add adjustment sliders
        self.setup_adjustment_sliders()
        
        # Create manipulation notebook
        self.manipulation_notebook = ctk.CTkTabview(self.control_frame, height=200)
        self.manipulation_notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # Add tabs for each manipulation category
        self.manipulation_notebook.add("Resize")
        self.manipulation_notebook.add("Flip")
        
        # Setup resize tab
        self.setup_resize_tab()
        
        # Setup flip tab
        self.setup_flip_tab()
        
        # Create effects notebook
        self.effects_notebook = ctk.CTkTabview(self.control_frame)
        self.effects_notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # Create tabs for each effect category
        for category in self.image_processor.effects.keys():
            self.effects_notebook.add(category)
            tab = self.effects_notebook.tab(category)
            
            # Add effect buttons for this category
            for effect_name in self.image_processor.effects[category].keys():
                effect_frame = ctk.CTkFrame(tab)
                effect_frame.pack(fill="x", padx=5, pady=2)
                
                effect_button = ctk.CTkButton(
                    effect_frame,
                    text=effect_name,
                    command=lambda e=effect_name, c=category: self.apply_effect(e, c)
                )
                effect_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
                
                # Add intensity slider
                intensity_slider = ctk.CTkSlider(
                    effect_frame,
                    from_=0,
                    to=1,
                    number_of_steps=100,
                    command=lambda v, e=effect_name, c=category: self.update_effect_intensity(e, c, v)
                )
                intensity_slider.set(1.0)
                intensity_slider.pack(side="left", fill="x", expand=True)

        # Create status bar
        self.status_bar = ctk.CTkLabel(self, text="Ready", anchor="w")
        self.status_bar.pack(fill="x", padx=10, pady=5)
        
    def setup_resize_tab(self):
        """Setup the resize tab with width and height inputs"""
        tab = self.manipulation_notebook.tab("Resize")
        
        # Width input
        width_frame = ctk.CTkFrame(tab)
        width_frame.pack(fill="x", padx=5, pady=5)
        
        width_label = ctk.CTkLabel(width_frame, text="Width:")
        width_label.pack(side="left", padx=5)
        
        self.width_entry = ctk.CTkEntry(width_frame, width=100)
        self.width_entry.pack(side="left", padx=5)
        self.width_entry.insert(0, "800")
        
        # Height input
        height_frame = ctk.CTkFrame(tab)
        height_frame.pack(fill="x", padx=5, pady=5)
        
        height_label = ctk.CTkLabel(height_frame, text="Height:")
        height_label.pack(side="left", padx=5)
        
        self.height_entry = ctk.CTkEntry(height_frame, width=100)
        self.height_entry.pack(side="left", padx=5)
        self.height_entry.insert(0, "600")
        
        # Resize button
        resize_button = ctk.CTkButton(
            tab,
            text="Resize Image",
            command=self.apply_resize
        )
        resize_button.pack(fill="x", padx=5, pady=10)
        
    def setup_flip_tab(self):
        """Setup the flip tab with direction selection"""
        tab = self.manipulation_notebook.tab("Flip")
        
        # Direction selection
        direction_frame = ctk.CTkFrame(tab)
        direction_frame.pack(fill="x", padx=5, pady=5)
        
        direction_label = ctk.CTkLabel(direction_frame, text="Direction:")
        direction_label.pack(side="left", padx=5)
        
        self.direction_var = ctk.StringVar(value="horizontal")
        
        horizontal_radio = ctk.CTkRadioButton(
            direction_frame,
            text="Horizontal",
            variable=self.direction_var,
            value="horizontal"
        )
        horizontal_radio.pack(side="left", padx=5)
        
        vertical_radio = ctk.CTkRadioButton(
            direction_frame,
            text="Vertical",
            variable=self.direction_var,
            value="vertical"
        )
        vertical_radio.pack(side="left", padx=5)
        
        # Flip button
        flip_button = ctk.CTkButton(
            tab,
            text="Flip Image",
            command=self.apply_flip
        )
        flip_button.pack(fill="x", padx=5, pady=10)
        
    def setup_adjustment_sliders(self):
        """Setup sliders for image adjustments"""
        # Brightness slider
        brightness_frame = ctk.CTkFrame(self.adjustments_frame)
        brightness_frame.pack(fill="x", pady=2)
        
        brightness_label = ctk.CTkLabel(brightness_frame, text="Brightness:")
        brightness_label.pack(side="left", padx=5)
        
        self.brightness_slider = ctk.CTkSlider(
            brightness_frame,
            from_=-100,
            to=100,
            variable=self.brightness_var,
            command=self.update_adjustments
        )
        self.brightness_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.brightness_slider.configure(state="disabled")
        
        # Contrast slider
        contrast_frame = ctk.CTkFrame(self.adjustments_frame)
        contrast_frame.pack(fill="x", pady=2)
        
        contrast_label = ctk.CTkLabel(contrast_frame, text="Contrast:")
        contrast_label.pack(side="left", padx=5)
        
        self.contrast_slider = ctk.CTkSlider(
            contrast_frame,
            from_=0.0,
            to=2.0,
            variable=self.contrast_var,
            command=self.update_adjustments
        )
        self.contrast_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.contrast_slider.configure(state="disabled")

    def update_adjustments(self, _=None):
        if self.image_processor.original_image is not None:
            brightness = self.brightness_var.get()
            contrast = self.contrast_var.get()

            # Use the original image as base for adjustment preview
            preview_image = self.image_processor.apply_adjustments_to_image(
                self.image_processor.current_image, brightness, contrast
            )
            self.update_image_display(preview_image)

    def apply_resize(self):
        """Apply resize operation based on user input"""
        try:
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())
            
            if width <= 0 or height <= 0:
                self.status_bar.configure(text="Error: Width and height must be positive numbers")
                return
                
            self.image_processor.resize_image(width, height)
            self.update_image_display()
            self.status_bar.configure(text=f"Image resized to {width}x{height}")
        except ValueError:
            self.status_bar.configure(text="Error: Please enter valid numbers for width and height")
            
    def apply_flip(self):
        """Apply flip operation based on user input"""
        direction = self.direction_var.get()
        
        self.image_processor.flip_image(direction)
        self.update_image_display()
        self.status_bar.configure(text=f"Image flipped {direction}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            try:
                self.image_processor.load_image(file_path)
                self.update_image_display()
                self.save_button.configure(state="normal")
                self.reset_button.configure(state="normal")
                self.format_combobox.configure(state="normal")
                
                # Enable adjustment sliders
                self.brightness_slider.configure(state="normal")
                self.contrast_slider.configure(state="normal")
                
                # Reset adjustment values
                self.brightness_var.set(0.0)
                self.contrast_var.set(1.0)
                
                self.status_bar.configure(text=f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                self.status_bar.configure(text=f"Error loading image: {str(e)}")

    def save_image(self):
        format_ext = self.format_var.get().lower()
        if format_ext == "jpeg":
            format_ext = "jpg"

        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{format_ext}",
            filetypes=[(f"{format_ext.upper()} Files", f"*.{format_ext}"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                # Apply brightness and contrast before saving
                brightness = self.brightness_var.get()
                contrast = self.contrast_var.get()

                adjusted = self.image_processor.apply_adjustments_to_image(
                    self.image_processor.current_image, brightness, contrast
                )
                cv2.imwrite(file_path, adjusted)
                self.status_bar.configure(text=f"Saved: {os.path.basename(file_path)}")
            except Exception as e:
                self.status_bar.configure(text=f"Error saving image: {str(e)}")

    def reset_image(self):
        self.image_processor.reset_image()
        self.update_image_display()
        
        # Reset adjustment values
        self.brightness_var.set(0.0)
        self.contrast_var.set(1.0)
        
        self.status_bar.configure(text="Image reset to original")

    def apply_effect(self, effect_name: str, category: str):
        effect_func = self.image_processor.effects[category][effect_name]
        self.image_processor.apply_effect(effect_func)
        self.update_image_display()
        self.status_bar.configure(text=f"Applied {effect_name} effect")

    def update_effect_intensity(self, effect_name: str, category: str, intensity: float):
        effect_func = self.image_processor.effects[category][effect_name]
        self.image_processor.apply_effect(effect_func, float(intensity))
        self.update_image_display()

    def update_image_display(self, image=None):
        if image is None:
            image = self.image_processor.current_image
            
        if image is not None:
            # Convert OpenCV image to PIL format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            
            # Get the display box dimensions
            display_width = self.image_frame.winfo_width()
            display_height = self.image_frame.winfo_height()
            
            # Get original image dimensions
            img_width, img_height = image.size
            
            # Calculate resize ratio to fit the display while maintaining aspect ratio
            width_ratio = display_width / img_width
            height_ratio = display_height / img_height
            resize_ratio = min(width_ratio, height_ratio)
            
            # Only resize if the image is larger than the display area
            if resize_ratio < 1.0:
                # Calculate new dimensions
                new_width = int(img_width * resize_ratio)
                new_height = int(img_height * resize_ratio)
                
                # Resize image
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.configure(image=photo, text="")  # Clear the text when image is loaded
            self.image_label.image = photo  # Keep a reference
            
            # Update histogram
            self.update_histogram()
        else:
            # If no image is loaded, show the placeholder text
            self.image_label.configure(image=None, text="No image loaded")
            self.clear_histogram()
            
    def update_histogram(self):
        """Update the histogram display with the current image's color distribution"""
        # Clear the previous plot
        self.hist_figure.clear()
        
        # Get histograms from the image processor
        hist_b, hist_g, hist_r = self.image_processor.calculate_histograms()
        
        if hist_b is not None:
            # Create the plot
            ax = self.hist_figure.add_subplot(111)
            
            # Plot histograms
            ax.plot(hist_b, color='blue', label='Blue', alpha=0.7)
            ax.plot(hist_g, color='green', label='Green', alpha=0.7)
            ax.plot(hist_r, color='red', label='Red', alpha=0.7)
            
            # Customize the plot
            ax.set_title('RGB Histogram')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
            
            # Adjust layout
            self.hist_figure.tight_layout()
            
            # Refresh the canvas here
            self.hist_canvas.draw()
            
    def clear_histogram(self):
        """Clear the histogram display"""
        self.hist_figure.clear()
        self.hist_canvas.draw()

if __name__ == "__main__":
    app = ImageEditorApp()
    app.mainloop()