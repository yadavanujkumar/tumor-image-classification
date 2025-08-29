"""
Data preprocessing module for medical imaging datasets.
Handles image loading, resizing, normalization, and augmentation.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Generator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Image processing will be limited.")

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logging.warning("Albumentations not available. Image augmentation will be disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Some image operations may fail.")

try:
    from ..config import PREPROCESSING_CONFIG, TRAINING_CONFIG
except ImportError:
    # Handle relative import issues for testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import PREPROCESSING_CONFIG, TRAINING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


class ImagePreprocessor:
    """Handles image preprocessing for medical imaging datasets."""
    
    def __init__(self, target_size: Tuple[int, int] = None, normalize: bool = True):
        self.target_size = target_size or PREPROCESSING_CONFIG['target_size']
        self.normalize = normalize
        self.label_encoder = LabelEncoder()
        
        # Setup augmentation pipeline
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
    
    def _create_train_transform(self) -> Any:
        """Create augmentation pipeline for training data."""
        if not ALBUMENTATIONS_AVAILABLE:
            logging.warning("Albumentations not available. Using basic transforms.")
            return None
        
        aug_config = PREPROCESSING_CONFIG['augmentation']
        
        transforms = [
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Rotate(limit=aug_config['rotation_range'], p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2, 
                scale_limit=0.2, 
                rotate_limit=aug_config['rotation_range'], 
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.normalize else A.NoOp()
        ]
        
        return A.Compose(transforms)
    
    def _create_val_transform(self) -> Any:
        """Create transformation pipeline for validation/test data."""
        if not ALBUMENTATIONS_AVAILABLE:
            logging.warning("Albumentations not available. Using basic transforms.")
            return None
        
        transforms = [
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.normalize else A.NoOp()
        ]
        
        return A.Compose(transforms)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                if CV2_AVAILABLE:
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise PreprocessingError(f"Could not load image: {image_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif PIL_AVAILABLE:
                    from PIL import Image as PILImage
                    image = PILImage.open(image_path)
                    image = np.array(image)
                else:
                    raise PreprocessingError("Neither OpenCV nor PIL available for image loading")
            else:
                image = image_path
            
            return image
            
        except Exception as e:
            raise PreprocessingError(f"Error loading image {image_path}: {str(e)}")
    
    def preprocess_image(self, image: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Preprocess a single image with augmentation.
        
        Args:
            image: Input image as numpy array
            is_training: Whether to apply training augmentations
            
        Returns:
            Preprocessed image
        """
        try:
            if ALBUMENTATIONS_AVAILABLE:
                transform = self.train_transform if is_training else self.val_transform
                if transform is not None:
                    augmented = transform(image=image)
                    return augmented['image']
            
            # Fallback: basic preprocessing without augmentation
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Simple resize fallback using basic numpy operations
                target_h, target_w = self.target_size
                current_h, current_w = image.shape[:2]
                
                # Simple resize (nearest neighbor for demo)
                resized = np.zeros((target_h, target_w, 3), dtype=image.dtype)
                h_ratio = current_h / target_h
                w_ratio = current_w / target_w
                
                for i in range(target_h):
                    for j in range(target_w):
                        orig_i = min(int(i * h_ratio), current_h - 1)
                        orig_j = min(int(j * w_ratio), current_w - 1)
                        resized[i, j] = image[orig_i, orig_j]
                
                if self.normalize:
                    resized = resized.astype(np.float32) / 255.0
                    # Basic normalization
                    resized = (resized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                
                return resized
            else:
                raise PreprocessingError("Unsupported image format")
            
        except Exception as e:
            raise PreprocessingError(f"Error preprocessing image: {str(e)}")
    
    def load_dataset_from_directory(self, data_dir: Path, class_names: List[str] = None) -> Tuple[List[str], List[str]]:
        """
        Load dataset from directory structure.
        
        Args:
            data_dir: Root directory containing class subdirectories
            class_names: List of class names (if None, inferred from directories)
            
        Returns:
            Tuple of (image_paths, labels)
        """
        data_dir = Path(data_dir)
        image_paths = []
        labels = []
        
        if class_names is None:
            class_names = [d.name for d in data_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Loading dataset from {data_dir}")
        logger.info(f"Found classes: {class_names}")
        
        for class_name in class_names:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory does not exist: {class_dir}")
                continue
            
            # Supported image extensions
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            
            class_images = []
            for ext in extensions:
                class_images.extend(class_dir.glob(f"*{ext}"))
                class_images.extend(class_dir.glob(f"*{ext.upper()}"))
            
            image_paths.extend([str(img) for img in class_images])
            labels.extend([class_name] * len(class_images))
            
            logger.info(f"Found {len(class_images)} images for class '{class_name}'")
        
        logger.info(f"Total images loaded: {len(image_paths)}")
        return image_paths, labels
    
    def create_dataset_splits(self, image_paths: List[str], labels: List[str], 
                            test_size: float = None, val_size: float = None, 
                            random_state: int = None) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            test_size: Proportion of test set
            val_size: Proportion of validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        test_size = test_size or TRAINING_CONFIG['test_split']
        val_size = val_size or TRAINING_CONFIG['validation_split']
        random_state = random_state or TRAINING_CONFIG['random_state']
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, encoded_labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=encoded_labels
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_adjusted, 
            random_state=random_state, 
            stratify=y_temp
        )
        
        # Convert labels back to strings
        y_train_str = self.label_encoder.inverse_transform(y_train).tolist()
        y_val_str = self.label_encoder.inverse_transform(y_val).tolist()
        y_test_str = self.label_encoder.inverse_transform(y_test).tolist()
        
        splits = {
            'train': (X_train, y_train_str),
            'val': (X_val, y_val_str),
            'test': (X_test, y_test_str)
        }
        
        # Log split information
        logger.info("Dataset splits created:")
        for split_name, (X, y) in splits.items():
            logger.info(f"  {split_name}: {len(X)} images")
            # Count class distribution
            unique, counts = np.unique(y, return_counts=True)
            for class_name, count in zip(unique, counts):
                logger.info(f"    {class_name}: {count} images")
        
        return splits
    
    def create_data_generator(self, image_paths: List[str], labels: List[str], 
                            batch_size: int = None, is_training: bool = False, 
                            shuffle: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Create a data generator for batch processing.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            batch_size: Batch size for generator
            is_training: Whether to apply training augmentations
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of (images, labels)
        """
        batch_size = batch_size or TRAINING_CONFIG['batch_size']
        
        # Encode labels
        if isinstance(labels[0], str):
            encoded_labels = self.label_encoder.transform(labels)
        else:
            encoded_labels = labels
        
        indices = np.arange(len(image_paths))
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    try:
                        # Load and preprocess image
                        image = self.load_image(image_paths[idx])
                        processed_image = self.preprocess_image(image, is_training=is_training)
                        
                        batch_images.append(processed_image)
                        batch_labels.append(encoded_labels[idx])
                        
                    except Exception as e:
                        logger.warning(f"Error processing image {image_paths[idx]}: {str(e)}")
                        continue
                
                if batch_images:
                    yield np.array(batch_images), np.array(batch_labels)
    
    def get_class_weights(self, labels: List[str]) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Args:
            labels: List of labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        unique_classes = np.unique(encoded_labels)
        
        class_weights = compute_class_weight(
            'balanced', 
            classes=unique_classes, 
            y=encoded_labels
        )
        
        return dict(zip(unique_classes, class_weights))
    
    def save_preprocessing_stats(self, save_path: Path, image_paths: List[str], labels: List[str]):
        """
        Save preprocessing statistics for reference.
        
        Args:
            save_path: Path to save statistics
            image_paths: List of image paths
            labels: List of labels
        """
        stats = {
            'total_images': len(image_paths),
            'target_size': self.target_size,
            'normalize': self.normalize,
            'class_distribution': {},
            'label_encoder_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else []
        }
        
        # Calculate class distribution
        unique, counts = np.unique(labels, return_counts=True)
        stats['class_distribution'] = dict(zip(unique, counts.tolist()))
        
        # Save to file
        import json
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Preprocessing statistics saved to {save_path}")


def main():
    """Main function for testing preprocessing."""
    from ..data.ingestion import DataIngestor
    
    # Setup sample dataset
    ingester = DataIngestor()
    dataset_info = ingester.setup_sample_dataset()
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # Load dataset (this will be empty for sample dataset)
    image_paths, labels = preprocessor.load_dataset_from_directory(
        dataset_info['path'], 
        dataset_info['classes']
    )
    
    print(f"Loaded {len(image_paths)} images")
    print(f"Classes: {dataset_info['classes']}")


if __name__ == "__main__":
    main()