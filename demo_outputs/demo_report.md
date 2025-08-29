# Tumor Image Classification - Demo Report

This is a demonstration of the tumor classification pipeline functionality.

## Model Performance Comparison

| Model        |   Accuracy |   Precision |   Recall |   F1-Score |   AUC-ROC |   Sensitivity |   Specificity |
|:-------------|-----------:|------------:|---------:|-----------:|----------:|--------------:|--------------:|
| EfficientNet |      0.920 |       0.944 |    0.911 |      0.927 |     0.930 |         0.911 |         0.932 |
| ResNet50     |      0.900 |       0.942 |    0.875 |      0.907 |     0.899 |         0.875 |         0.932 |
| Custom_CNN   |      0.850 |       0.860 |    0.875 |      0.867 |     0.824 |         0.875 |         0.818 |

## Best Model Analysis

**Best Model:** EfficientNet

- **Accuracy:** 0.920
- **Sensitivity:** 0.911
- **Specificity:** 0.932

## Clinical Interpretation

✓ **Excellent sensitivity** - Low risk of missing malignant cases

## Generated Files

- Model comparison: `demo_outputs/model_comparison.png`
- Confusion matrices: `demo_outputs/*_confusion_matrix.png`
- ROC curves: `demo_outputs/*_roc_curve.png`
- Class distribution: `demo_outputs/class_distribution.png`
