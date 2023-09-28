import os
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.experiment.utils import calculate_iou

@dataclass
class SegmentationOutput:
    iou: float
    image: np.ndarray
    prediction: np.ndarray
    ground_truth: np.ndarray
    image_path: str
    ground_truth_path: str

    def plot_segmentation(self, alpha=0.5) -> None:
        # Create a copy of the original image to avoid modifying it
        result1 = self.image.copy()
        result2 = self.image.copy()

        # Convert the mask to a 3-channel format (if it's single-channel)
        if len(self.prediction.shape) == 2:
            mask = cv2.cvtColor(self.prediction.copy(), cv2.COLOR_GRAY2BGR)

                # Convert the mask to a 3-channel format (if it's single-channel)
        if len(self.ground_truth.shape) == 2:
            ground_truth = cv2.cvtColor(self.ground_truth.copy(), cv2.COLOR_GRAY2BGR)

        # Convert masks to yellow
        mask[:, :, 0] = 0
        mask[:, :, 1][mask[:, :, 1]>0] = 255
        mask[:, :, 2][mask[:, :, 2]>0] = 255
        # Convert ground truth to green
        ground_truth[:, :, 0] = 0
        ground_truth[:, :, 1][ground_truth[:, :, 1]>0] = 255
        ground_truth[:, :, 2] = 0

        # Blend the mask and the original image
        cv2.addWeighted(mask, alpha, result1, 1 - alpha, 0, result1)
        cv2.addWeighted(ground_truth, alpha, result2, 1 - alpha, 0, result2)

        fig, axes = plt.subplots(1, 2)
        plt.suptitle(f"IoU: {self.iou*100:.2f}%")

        axes[0].set_title("Prediction")
        axes[0].imshow(result1)
        axes[1].set_title("Ground Truth")
        axes[1].imshow(result2)

        axes[0].axis("off")
        axes[1].axis("off")


class SegmentationPipeline:
    def __init__(
            self,
            blur_kernel_size: int = 5,
            blur_sigma: int = 0,
            canny_threshold_1: int = 40,
            canny_threshold_2: int = 120,
            clip_limit: float = 2.0,
            tile_grid_size: int = 8,
            brightness_alpha: float = 1.0,
            brightness_beta: int = 0,
            pre_blur: bool = False,
            sharpen_mode: str = None,
            kernel_size: int = 9,
            sigma_x: int = 10,
            sigma_y: int = 10,
            **kwargs
    ) -> None:
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.canny_threshold_1 = canny_threshold_1
        self.canny_threshold_2 = canny_threshold_2
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.brightness_alpha = brightness_alpha
        self.brightness_beta = brightness_beta
        self.pre_blur = pre_blur
        self.sharpen_mode = sharpen_mode
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def _get_contours(self, img_path: str) -> tuple[np.ndarray, tuple[np.ndarray]]:
        """
        Get the contours of the objects in the image.

        Args:
            img_path (str): The path to the image.

        Returns:
            np.ndarray: The contours of the objects in the image.
        """
        # Load the input image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Adjust brightness and contrast
        adjusted_image = cv2.convertScaleAbs(image, alpha=self.brightness_alpha, beta=self.brightness_beta)

        # Convert the adjusted image to grayscale
        gray = cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2GRAY)

        # Enhance the grayscale image
        enhanced_gray = self._enhance_gray_image(gray.copy())

        # Apply Canny edge detection
        edges = cv2.Canny(enhanced_gray, self.canny_threshold_1, self.canny_threshold_2)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return image, contours

    def _enhance_gray_image(self, image: np.ndarray) -> np.ndarray:
        # Apply contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_grid_size, self.tile_grid_size))
        enhanced_gray = clahe.apply(image)

        # Apply Gaussian blur to reduce noise
        if self.pre_blur:
            enhanced_gray = cv2.GaussianBlur(enhanced_gray.copy(), (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma)
            enhanced_gray = self._apply_sharpening(enhanced_gray.copy())
        else: 
            enhanced_gray = self._apply_sharpening(enhanced_gray.copy())
            enhanced_gray = cv2.GaussianBlur(enhanced_gray.copy(), (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma)
        
        return enhanced_gray

    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        if self.sharpen_mode is None:
            sharpened = image
        elif self.sharpen_mode == "default":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
        elif self.sharpen_mode == "gaussian":
            sharpened = cv2.GaussianBlur(image, (0, 0), 3)
            sharpened = cv2.addWeighted(sharpened, 1.5, sharpened, -0.5, 0)
        elif self.sharpen_mode == "laplacian":
            sharpened = cv2.Laplacian(image, -1, ksize=3)
        else:
            raise ValueError(f"Unknown sharpening mode: {self.sharpen_mode}")

        return sharpened
    
    def _create_coin_mask(self, image: np.ndarray, contours: tuple[np.ndarray]) -> np.ndarray:
        # Create an empty black image with the same dimensions as the input image
        mask = np.zeros_like(image)

        # Draw the filled contours on the mask
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # Convert the mask to grayscale
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Threshold the mask to create a binary mask
        _, binary_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

        smoothed_mask = cv2.GaussianBlur(binary_mask, (self.kernel_size,self.kernel_size), sigmaX=self.sigma_x, sigmaY=self.sigma_y)

        return smoothed_mask
    
    def _fill_inner_mask(self, smoothed_mask: np.ndarray) -> np.ndarray:
        # Find contours in the smoothed mask
        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty black mask with the same dimensions as the smoothed mask
        filled_mask = np.zeros_like(smoothed_mask)

        # Fill the inner contours
        for contour in contours:
            cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

        return filled_mask
    
    def segment(self, images: np.array, masks: np.array) -> list[SegmentationOutput]:
        outputs = []
        for img_path, mask_path in zip(images, masks):     
            ground_truth = cv2.imread(mask_path).mean(axis=-1).astype(np.uint8)
            image, contours = self._get_contours(img_path)
            binary_mask = self._create_coin_mask(image, contours)
            prediction = self._fill_inner_mask(binary_mask)
            iou = calculate_iou(ground_truth, prediction)
            outputs.append(SegmentationOutput(iou, image, prediction, ground_truth, img_path, mask_path))

        return outputs