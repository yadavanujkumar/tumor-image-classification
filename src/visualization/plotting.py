"""
Visualization module for tumor classification results.
Includes confusion matrices, ROC curves, and Grad-CAM heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

# Optional imports
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Some model-specific visualizations will be disabled.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Some image operations may be limited.")

# Try to import grad-cam
try:
    from grad_cam import GradCAM
    from grad_cam.utils.image import show_cam_on_image, preprocess_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logging.warning("grad-cam not available. Grad-CAM visualizations will be disabled.")

try:
    from ..config import VISUALIZATION_CONFIG
except ImportError:
    # Handle relative import issues for testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import VISUALIZATION_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass


class TumorClassificationVisualizer:
    """Visualization tools for tumor classification analysis."""
    
    def __init__(self, class_names: List[str], figsize: Tuple[int, int] = None):
        self.class_names = class_names
        self.figsize = figsize or VISUALIZATION_CONFIG['figsize']
        
        # Set style
        plt.style.use('default')  # Use default since seaborn-v0_8 might not be available
        sns.set_palette(VISUALIZATION_CONFIG['color_palette'])
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            title: str = "Confusion Matrix",
                            save_path: Path = None, 
                            normalize: bool = False) -> plt.Figure:
        """
        Plot confusion matrix with customization options.
        
        Args:
            confusion_matrix: Confusion matrix array
            title: Plot title
            save_path: Path to save the plot
            normalize: Whether to normalize the matrix
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            if normalize:
                cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
                title += " (Normalized)"
            else:
                sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                           xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error plotting confusion matrix: {str(e)}")
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, 
                      auc_score: float, title: str = "ROC Curve",
                      save_path: Path = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc_score: AUC score
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {auc_score:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
                logger.info(f"ROC curve saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error plotting ROC curve: {str(e)}")
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            title: str = "Training History",
                            save_path: Path = None) -> plt.Figure:
        """
        Plot training history (loss and metrics).
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            # Determine number of subplots based on available metrics
            metrics = [key for key in history.keys() if not key.startswith('val_')]
            n_metrics = len(metrics)
            
            fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                # Plot training metric
                ax.plot(history[metric], label=f'Training {metric}', linewidth=2)
                
                # Plot validation metric if available
                val_metric = f'val_{metric}'
                if val_metric in history:
                    ax.plot(history[val_metric], label=f'Validation {metric}', linewidth=2)
                
                ax.set_title(f'{metric.capitalize()}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(metric.capitalize(), fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
                logger.info(f"Training history saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error plotting training history: {str(e)}")
    
    def plot_class_distribution(self, labels: List[str], 
                              title: str = "Class Distribution",
                              save_path: Path = None) -> plt.Figure:
        """
        Plot class distribution bar chart.
        
        Args:
            labels: List of labels
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Count occurrences of each class
            unique, counts = np.unique(labels, return_counts=True)
            
            # Create bar plot
            bars = ax.bar(unique, counts, color=sns.color_palette("viridis", len(unique)))
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=12)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel('Number of Images', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Calculate percentages
            total = sum(counts)
            percentages = [f'{count/total*100:.1f}%' for count in counts]
            
            # Add percentage labels
            for i, (bar, percentage) in enumerate(zip(bars, percentages)):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                       percentage, ha='center', va='center', fontsize=10, 
                       color='white', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
                logger.info(f"Class distribution saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error plotting class distribution: {str(e)}")
    
    def plot_model_comparison(self, comparison_df, 
                            title: str = "Model Performance Comparison",
                            save_path: Path = None) -> plt.Figure:
        """
        Plot model comparison chart.
        
        Args:
            comparison_df: DataFrame with model comparison results
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            # Select numeric columns for comparison
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'Model']
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create grouped bar chart
            x = np.arange(len(comparison_df))
            width = 0.8 / len(numeric_cols)
            
            for i, metric in enumerate(numeric_cols):
                offset = (i - len(numeric_cols)/2) * width + width/2
                bars = ax.bar(x + offset, comparison_df[metric], width, 
                            label=metric, alpha=0.8)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
                logger.info(f"Model comparison saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error plotting model comparison: {str(e)}")
    
    def generate_gradcam_heatmap(self, model, image: np.ndarray, 
                               class_index: int, layer_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap for model interpretability.
        
        Args:
            model: Keras model (requires TensorFlow)
            image: Input image (preprocessed)
            class_index: Target class index
            layer_name: Name of layer for Grad-CAM (if None, uses last conv layer)
            
        Returns:
            Tuple of (heatmap, superimposed_image)
        """
        if not GRADCAM_AVAILABLE:
            logger.warning("Grad-CAM not available. Install grad-cam package.")
            return None, None
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Grad-CAM requires TensorFlow.")
            return None, None
        
        try:
            # Find last convolutional layer if not specified
            if layer_name is None:
                for layer in reversed(model.layers):
                    if 'conv' in layer.name.lower():
                        layer_name = layer.name
                        break
                
                if layer_name is None:
                    raise VisualizationError("No convolutional layer found for Grad-CAM")
            
            # Initialize Grad-CAM
            cam = GradCAM(model=model, classIdx=class_index, layerName=layer_name)
            
            # Generate heatmap
            heatmap = cam.compute_heatmap(image)
            
            # Normalize image for visualization
            if image.max() <= 1.0:
                vis_image = (image * 255).astype(np.uint8)
            else:
                vis_image = image.astype(np.uint8)
            
            # Superimpose heatmap on image
            superimposed = show_cam_on_image(vis_image / 255.0, heatmap, use_rgb=True)
            
            return heatmap, superimposed
            
        except Exception as e:
            raise VisualizationError(f"Error generating Grad-CAM: {str(e)}")
    
    def plot_gradcam_analysis(self, model, images: List[np.ndarray], 
                            true_labels: List[int], predictions: List[int],
                            image_paths: List[str] = None, n_samples: int = 6,
                            save_path: Path = None) -> plt.Figure:
        """
        Plot Grad-CAM analysis for multiple images.
        
        Args:
            model: Keras model (requires TensorFlow)
            images: List of preprocessed images
            true_labels: List of true labels
            predictions: List of predicted labels
            image_paths: List of image paths for titles
            n_samples: Number of samples to display
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not GRADCAM_AVAILABLE or not TENSORFLOW_AVAILABLE:
            logger.warning("Grad-CAM or TensorFlow not available. Skipping Grad-CAM analysis.")
            return None
        
        try:
            n_samples = min(n_samples, len(images))
            fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
            
            if n_samples == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(n_samples):
                image = images[i]
                true_label = true_labels[i]
                pred_label = predictions[i]
                
                # Original image
                if image.max() <= 1.0:
                    display_image = (image * 255).astype(np.uint8)
                else:
                    display_image = image.astype(np.uint8)
                
                axes[0, i].imshow(display_image)
                axes[0, i].set_title(f'True: {self.class_names[true_label]}\\nPred: {self.class_names[pred_label]}',
                                   fontsize=10)
                axes[0, i].axis('off')
                
                # Grad-CAM heatmap
                try:
                    heatmap, superimposed = self.generate_gradcam_heatmap(
                        model, np.expand_dims(image, axis=0), pred_label
                    )
                    
                    if superimposed is not None:
                        axes[1, i].imshow(superimposed)
                        axes[1, i].set_title('Grad-CAM Heatmap', fontsize=10)
                    else:
                        axes[1, i].text(0.5, 0.5, 'Grad-CAM\\nNot Available', 
                                      ha='center', va='center', transform=axes[1, i].transAxes)
                except Exception as e:
                    axes[1, i].text(0.5, 0.5, f'Error:\\n{str(e)[:20]}...', 
                                  ha='center', va='center', transform=axes[1, i].transAxes)
                
                axes[1, i].axis('off')
            
            plt.suptitle('Grad-CAM Analysis: Model Attention Regions', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
                logger.info(f"Grad-CAM analysis saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error plotting Grad-CAM analysis: {str(e)}")
    
    def create_comprehensive_report(self, results: Dict[str, Any], 
                                  model_name: str, save_dir: Path) -> Dict[str, Path]:
        """
        Create comprehensive visualization report.
        
        Args:
            results: Evaluation results dictionary
            model_name: Name of the model
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary with paths to saved visualizations
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        try:
            # Confusion matrix
            if 'confusion_matrix' in results:
                cm_path = save_dir / f"{model_name}_confusion_matrix.png"
                fig = self.plot_confusion_matrix(
                    results['confusion_matrix'], 
                    title=f"{model_name} - Confusion Matrix",
                    save_path=cm_path
                )
                plt.close(fig)
                saved_plots['confusion_matrix'] = cm_path
                
                # Normalized confusion matrix
                cm_norm_path = save_dir / f"{model_name}_confusion_matrix_normalized.png"
                fig = self.plot_confusion_matrix(
                    results['confusion_matrix'], 
                    title=f"{model_name} - Normalized Confusion Matrix",
                    save_path=cm_norm_path,
                    normalize=True
                )
                plt.close(fig)
                saved_plots['confusion_matrix_normalized'] = cm_norm_path
            
            # ROC curve
            if 'roc_curve' in results:
                roc_path = save_dir / f"{model_name}_roc_curve.png"
                fig = self.plot_roc_curve(
                    results['roc_curve']['fpr'],
                    results['roc_curve']['tpr'],
                    results['auc_roc'],
                    title=f"{model_name} - ROC Curve",
                    save_path=roc_path
                )
                plt.close(fig)
                saved_plots['roc_curve'] = roc_path
            
            logger.info(f"Comprehensive visualization report created in {save_dir}")
            return saved_plots
            
        except Exception as e:
            raise VisualizationError(f"Error creating comprehensive report: {str(e)}")


def main():
    """Main function for testing visualizations."""
    # Create dummy data for testing
    visualizer = TumorClassificationVisualizer(['benign', 'malignant'])
    
    # Test confusion matrix
    cm = np.array([[85, 15], [10, 90]])
    fig = visualizer.plot_confusion_matrix(cm, "Test Confusion Matrix")
    plt.show()
    
    # Test ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Dummy ROC curve
    auc_score = 0.85
    fig = visualizer.plot_roc_curve(fpr, tpr, auc_score, "Test ROC Curve")
    plt.show()
    
    # Test class distribution
    labels = ['benign'] * 80 + ['malignant'] * 120
    fig = visualizer.plot_class_distribution(labels, "Test Class Distribution")
    plt.show()


if __name__ == "__main__":
    main()