"""
Model evaluation module for tumor classification.
Provides comprehensive metrics and performance analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelBinarizer
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path

# Optional imports
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Some functionality may be limited.")

try:
    from ..config import METRICS
except ImportError:
    # Handle relative import issues for testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import METRICS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass


class ModelEvaluator:
    """Comprehensive model evaluation for tumor classification."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(range(self.num_classes))
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Evaluate model predictions with comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        try:
            results = {}
            
            # Basic classification metrics
            results['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Handle binary vs multiclass
            if self.num_classes == 2:
                # Binary classification
                results['precision'] = precision_score(y_true, y_pred, average='binary')
                results['recall'] = recall_score(y_true, y_pred, average='binary')
                results['f1_score'] = f1_score(y_true, y_pred, average='binary')
                
                if y_pred_proba is not None:
                    # For binary classification, use probability of positive class
                    if y_pred_proba.shape[1] == 2:
                        proba_positive = y_pred_proba[:, 1]
                    else:
                        proba_positive = y_pred_proba.flatten()
                    results['auc_roc'] = roc_auc_score(y_true, proba_positive)
                    
                    # Store ROC curve data
                    fpr, tpr, thresholds = roc_curve(y_true, proba_positive)
                    results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
            else:
                # Multiclass classification
                results['precision'] = precision_score(y_true, y_pred, average='weighted')
                results['recall'] = recall_score(y_true, y_pred, average='weighted')
                results['f1_score'] = f1_score(y_true, y_pred, average='weighted')
                
                if y_pred_proba is not None:
                    # Multiclass ROC AUC
                    y_true_binary = self.label_binarizer.transform(y_true)
                    results['auc_roc'] = roc_auc_score(y_true_binary, y_pred_proba, average='weighted')
            
            # Per-class metrics
            results['per_class_precision'] = precision_score(y_true, y_pred, average=None)
            results['per_class_recall'] = recall_score(y_true, y_pred, average=None)
            results['per_class_f1'] = f1_score(y_true, y_pred, average=None)
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            
            # Classification report
            results['classification_report'] = classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True
            )
            
            # Additional metrics for medical imaging
            results['sensitivity'] = results['recall']  # Same as recall
            specificity = self._calculate_specificity(y_true, y_pred)
            results['specificity'] = specificity
            
            logger.info("Model evaluation completed successfully")
            return results
            
        except Exception as e:
            raise EvaluationError(f"Error during evaluation: {str(e)}")
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        if self.num_classes == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # For multiclass, calculate average specificity
            cm = confusion_matrix(y_true, y_pred)
            specificity_per_class = []
            
            for i in range(self.num_classes):
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificity_per_class.append(specificity)
            
            return np.mean(specificity_per_class)
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models' performance.
        
        Args:
            model_results: Dictionary with model names as keys and evaluation results as values
            
        Returns:
            DataFrame comparing model performances
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1_score', 0),
                'AUC-ROC': results.get('auc_roc', 0),
                'Sensitivity': results.get('sensitivity', 0),
                'Specificity': results.get('specificity', 0)
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        logger.info("Model comparison completed")
        return comparison_df
    
    def calculate_statistical_significance(self, results1: Dict[str, Any], 
                                         results2: Dict[str, Any], 
                                         n_samples: int) -> Dict[str, float]:
        """
        Calculate statistical significance between two models using McNemar's test.
        
        Args:
            results1: Results from first model
            results2: Results from second model
            n_samples: Number of test samples
            
        Returns:
            Dictionary with p-values for different metrics
        """
        try:
            from scipy.stats import mcnemar
            
            # This is a simplified implementation
            # In practice, you'd need the actual predictions for McNemar's test
            significance_results = {
                'accuracy_diff': abs(results1['accuracy'] - results2['accuracy']),
                'f1_diff': abs(results1['f1_score'] - results2['f1_score']),
                'note': 'Statistical significance test requires actual predictions for proper implementation'
            }
            
            return significance_results
            
        except ImportError:
            logger.warning("scipy not available for statistical tests")
            return {'note': 'scipy required for statistical significance testing'}
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 model_name: str = "Model") -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report as string
        """
        report = f"\n{'='*60}\n"
        report += f"EVALUATION REPORT: {model_name}\n"
        report += f"{'='*60}\n\n"
        
        # Overall performance
        report += "OVERALL PERFORMANCE:\n"
        report += f"Accuracy:     {results['accuracy']:.4f}\n"
        report += f"Precision:    {results['precision']:.4f}\n"
        report += f"Recall:       {results['recall']:.4f}\n"
        report += f"F1-Score:     {results['f1_score']:.4f}\n"
        
        if 'auc_roc' in results:
            report += f"AUC-ROC:      {results['auc_roc']:.4f}\n"
        
        report += f"Sensitivity:  {results['sensitivity']:.4f}\n"
        report += f"Specificity:  {results['specificity']:.4f}\n\n"
        
        # Per-class performance
        if 'per_class_precision' in results:
            report += "PER-CLASS PERFORMANCE:\n"
            for i, class_name in enumerate(self.class_names):
                if i < len(results['per_class_precision']):
                    report += f"{class_name}:\n"
                    report += f"  Precision: {results['per_class_precision'][i]:.4f}\n"
                    report += f"  Recall:    {results['per_class_recall'][i]:.4f}\n"
                    report += f"  F1-Score:  {results['per_class_f1'][i]:.4f}\n"
            report += "\n"
        
        # Confusion matrix
        if 'confusion_matrix' in results:
            report += "CONFUSION MATRIX:\n"
            cm = results['confusion_matrix']
            
            # Header
            report += "           "
            for class_name in self.class_names:
                report += f"{class_name:>10}"
            report += "\n"
            
            # Matrix rows
            for i, class_name in enumerate(self.class_names):
                report += f"{class_name:>10}"
                for j in range(len(self.class_names)):
                    if i < cm.shape[0] and j < cm.shape[1]:
                        report += f"{cm[i, j]:>10}"
                    else:
                        report += f"{'0':>10}"
                report += "\n"
            report += "\n"
        
        # Medical relevance interpretation
        report += "MEDICAL RELEVANCE:\n"
        report += self._interpret_medical_metrics(results)
        
        return report
    
    def _interpret_medical_metrics(self, results: Dict[str, Any]) -> str:
        """
        Provide medical interpretation of the metrics.
        
        Args:
            results: Evaluation results
            
        Returns:
            Medical interpretation text
        """
        interpretation = ""
        
        sensitivity = results.get('sensitivity', 0)
        specificity = results.get('specificity', 0)
        
        interpretation += f"• Sensitivity (Recall): {sensitivity:.4f}\n"
        interpretation += "  - Ability to correctly identify malignant tumors\n"
        interpretation += "  - High sensitivity reduces false negatives (missed cancers)\n\n"
        
        interpretation += f"• Specificity: {specificity:.4f}\n"
        interpretation += "  - Ability to correctly identify benign tumors\n"
        interpretation += "  - High specificity reduces false positives (unnecessary procedures)\n\n"
        
        # Provide clinical context
        if sensitivity >= 0.95:
            interpretation += "• EXCELLENT sensitivity - Very low risk of missing malignant cases\n"
        elif sensitivity >= 0.90:
            interpretation += "• GOOD sensitivity - Acceptable for clinical screening\n"
        elif sensitivity >= 0.80:
            interpretation += "• MODERATE sensitivity - May miss some malignant cases\n"
        else:
            interpretation += "• LOW sensitivity - High risk of missing malignant cases\n"
        
        if specificity >= 0.95:
            interpretation += "• EXCELLENT specificity - Very low false positive rate\n"
        elif specificity >= 0.90:
            interpretation += "• GOOD specificity - Acceptable false positive rate\n"
        elif specificity >= 0.80:
            interpretation += "• MODERATE specificity - Some unnecessary procedures\n"
        else:
            interpretation += "• LOW specificity - High false positive rate\n"
        
        return interpretation
    
    def save_results(self, results: Dict[str, Any], save_path: Path, model_name: str = "model"):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            save_path: Path to save results
            model_name: Name of the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        import json
        results_copy = results.copy()
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in results_copy.items():
            if isinstance(value, np.ndarray):
                results_copy[key] = value.tolist()
            elif key == 'roc_curve' and isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        results_copy[key][subkey] = subvalue.tolist()
        
        json_path = save_path / f"{model_name}_evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        # Save formatted report
        report = self.generate_evaluation_report(results, model_name)
        report_path = save_path / f"{model_name}_evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation results saved to {save_path}")


def main():
    """Main function for testing evaluation."""
    # Create dummy data for testing
    evaluator = ModelEvaluator(['benign', 'malignant'])
    
    # Simulate some predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100, 2)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Evaluate
    results = evaluator.evaluate_predictions(y_true, y_pred, y_pred_proba)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results, "Test Model")
    print(report)


if __name__ == "__main__":
    main()