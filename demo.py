"""
Demo script for tumor classification pipeline.
This script demonstrates the core functionality without requiring TensorFlow/heavy dependencies.
"""

import sys
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

# Import pipeline components
from data.ingestion import DataIngestor
from data.preprocessing import ImagePreprocessor
from evaluation.metrics import ModelEvaluator
from visualization.plotting import TumorClassificationVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_data_ingestion():
    """Demonstrate data ingestion functionality."""
    print("\n" + "="*60)
    print("1. DATA INGESTION DEMO")
    print("="*60)
    
    ingester = DataIngestor()
    
    # Setup sample dataset
    dataset_info = ingester.setup_sample_dataset()
    print(f"✓ Sample dataset created: {dataset_info['name']}")
    print(f"  Path: {dataset_info['path']}")
    print(f"  Classes: {dataset_info['classes']}")
    
    # List available datasets
    available = ingester.list_available_datasets()
    print(f"\n✓ Available datasets:")
    for name, info in available.items():
        status = "✓" if info['exists'] else "✗"
        print(f"  {status} {name}: {info['path']}")
    
    return dataset_info


def demo_preprocessing():
    """Demonstrate preprocessing functionality."""
    print("\n" + "="*60)
    print("2. DATA PREPROCESSING DEMO")
    print("="*60)
    
    preprocessor = ImagePreprocessor()
    
    # Create mock image paths and labels for demonstration
    image_paths = [f"mock_image_{i}.jpg" for i in range(100)]
    labels = ['benign'] * 60 + ['malignant'] * 40
    
    print(f"✓ Mock dataset: {len(image_paths)} images")
    print(f"  Classes: {set(labels)}")
    
    # Create dataset splits
    splits = preprocessor.create_dataset_splits(image_paths, labels)
    print(f"\n✓ Dataset splits created:")
    for split_name, (paths, lbls) in splits.items():
        print(f"  {split_name}: {len(paths)} images")
    
    # Calculate class weights
    class_weights = preprocessor.get_class_weights(labels)
    print(f"\n✓ Class weights: {class_weights}")
    
    return splits, class_weights


def demo_evaluation():
    """Demonstrate evaluation functionality."""
    print("\n" + "="*60)
    print("3. MODEL EVALUATION DEMO")
    print("="*60)
    
    evaluator = ModelEvaluator(['benign', 'malignant'])
    
    # Create mock predictions
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    
    # Simulate different model performances
    models_results = {}
    
    for model_name, base_acc in [('ResNet50', 0.90), ('EfficientNet', 0.92), ('Custom_CNN', 0.85)]:
        # Create predictions with some noise around base accuracy
        correct_predictions = int(base_acc * n_samples)
        y_pred = y_true.copy()
        
        # Introduce some errors
        error_indices = np.random.choice(n_samples, n_samples - correct_predictions, replace=False)
        y_pred[error_indices] = 1 - y_pred[error_indices]
        
        # Create probability predictions
        y_pred_proba = np.random.rand(n_samples, 2)
        for i in range(n_samples):
            if y_pred[i] == 1:
                y_pred_proba[i] = [0.3, 0.7] + np.random.normal(0, 0.1, 2)
            else:
                y_pred_proba[i] = [0.7, 0.3] + np.random.normal(0, 0.1, 2)
        
        # Normalize probabilities
        y_pred_proba = np.abs(y_pred_proba)
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
        
        # Evaluate model
        results = evaluator.evaluate_predictions(y_true, y_pred, y_pred_proba)
        models_results[model_name] = results
        
        print(f"\n✓ {model_name} Results:")
        print(f"  Accuracy: {results['accuracy']:.3f}")
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall: {results['recall']:.3f}")
        print(f"  F1-Score: {results['f1_score']:.3f}")
        print(f"  AUC-ROC: {results['auc_roc']:.3f}")
    
    # Compare models
    comparison_df = evaluator.compare_models(models_results)
    print(f"\n✓ Model Comparison:")
    print(comparison_df.to_string(index=False, float_format='%.3f'))
    
    return models_results, comparison_df


def demo_visualization(models_results, comparison_df):
    """Demonstrate visualization functionality."""
    print("\n" + "="*60)
    print("4. VISUALIZATION DEMO")
    print("="*60)
    
    visualizer = TumorClassificationVisualizer(['benign', 'malignant'])
    
    # Create output directory
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Model comparison plot
    comp_path = output_dir / "model_comparison.png"
    fig = visualizer.plot_model_comparison(
        comparison_df, 
        "Model Performance Comparison",
        save_path=comp_path
    )
    plt.close(fig)
    print(f"✓ Model comparison plot saved: {comp_path}")
    
    # 2. Confusion matrices for each model
    for model_name, results in models_results.items():
        cm_path = output_dir / f"{model_name}_confusion_matrix.png"
        fig = visualizer.plot_confusion_matrix(
            results['confusion_matrix'],
            f"{model_name} - Confusion Matrix",
            save_path=cm_path
        )
        plt.close(fig)
        print(f"✓ Confusion matrix saved: {cm_path}")
        
        # ROC curve
        if 'roc_curve' in results:
            roc_path = output_dir / f"{model_name}_roc_curve.png"
            fig = visualizer.plot_roc_curve(
                results['roc_curve']['fpr'],
                results['roc_curve']['tpr'],
                results['auc_roc'],
                f"{model_name} - ROC Curve",
                save_path=roc_path
            )
            plt.close(fig)
            print(f"✓ ROC curve saved: {roc_path}")
    
    # 3. Class distribution plot
    labels = ['benign'] * 60 + ['malignant'] * 40
    dist_path = output_dir / "class_distribution.png"
    fig = visualizer.plot_class_distribution(
        labels,
        "Dataset Class Distribution",
        save_path=dist_path
    )
    plt.close(fig)
    print(f"✓ Class distribution plot saved: {dist_path}")
    
    print(f"\n✓ All visualizations saved in: {output_dir}")
    return output_dir


def generate_demo_report(models_results, comparison_df, output_dir):
    """Generate a demo report."""
    print("\n" + "="*60)
    print("5. REPORT GENERATION DEMO")
    print("="*60)
    
    report_path = output_dir / "demo_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Tumor Image Classification - Demo Report\n\n")
        f.write("This is a demonstration of the tumor classification pipeline functionality.\n\n")
        
        f.write("## Model Performance Comparison\n\n")
        f.write(comparison_df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n")
        
        f.write("## Best Model Analysis\n\n")
        best_model = comparison_df.iloc[0]['Model']
        best_results = models_results[best_model]
        
        f.write(f"**Best Model:** {best_model}\n\n")
        f.write(f"- **Accuracy:** {best_results['accuracy']:.3f}\n")
        f.write(f"- **Sensitivity:** {best_results['sensitivity']:.3f}\n")
        f.write(f"- **Specificity:** {best_results['specificity']:.3f}\n\n")
        
        f.write("## Clinical Interpretation\n\n")
        sensitivity = best_results['sensitivity']
        if sensitivity >= 0.90:
            f.write("✓ **Excellent sensitivity** - Low risk of missing malignant cases\n")
        elif sensitivity >= 0.80:
            f.write("⚠ **Good sensitivity** - Acceptable for clinical screening\n")
        else:
            f.write("⚠ **Moderate sensitivity** - May miss some malignant cases\n")
        
        f.write("\n## Generated Files\n\n")
        f.write(f"- Model comparison: `{output_dir}/model_comparison.png`\n")
        f.write(f"- Confusion matrices: `{output_dir}/*_confusion_matrix.png`\n")
        f.write(f"- ROC curves: `{output_dir}/*_roc_curve.png`\n")
        f.write(f"- Class distribution: `{output_dir}/class_distribution.png`\n")
    
    print(f"✓ Demo report generated: {report_path}")
    return report_path


def main():
    """Run the complete demo."""
    print("🚀 TUMOR IMAGE CLASSIFICATION PIPELINE DEMO")
    print("This demo showcases the pipeline functionality with mock data.")
    
    try:
        # 1. Data Ingestion
        dataset_info = demo_data_ingestion()
        
        # 2. Preprocessing
        splits, class_weights = demo_preprocessing()
        
        # 3. Evaluation
        models_results, comparison_df = demo_evaluation()
        
        # 4. Visualization
        output_dir = demo_visualization(models_results, comparison_df)
        
        # 5. Report Generation
        report_path = generate_demo_report(models_results, comparison_df, output_dir)
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Results saved in: {output_dir}")
        print(f"📋 Report available at: {report_path}")
        print("\nTo run with real data and models, install TensorFlow and use:")
        print("  python main.py --dataset sample --models custom_cnn resnet50")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)