"""
Test script to verify the tumor classification pipeline works correctly.
This script runs a minimal version of the pipeline for testing purposes.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import pipeline components
from main import TumorClassificationPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pipeline_basic():
    """Test basic pipeline functionality."""
    logger.info("Starting basic pipeline test")
    
    try:
        # Initialize pipeline
        pipeline = TumorClassificationPipeline()
        
        # Test data ingestion
        dataset_info = pipeline.run_data_ingestion("sample")
        assert dataset_info is not None
        logger.info("✓ Data ingestion test passed")
        
        # Test exploratory analysis
        analysis_results = pipeline.run_exploratory_analysis(dataset_info)
        assert analysis_results is not None
        logger.info("✓ Exploratory analysis test passed")
        
        # Test preprocessing
        preprocessing_results = pipeline.run_data_preprocessing(analysis_results)
        assert preprocessing_results is not None
        logger.info("✓ Data preprocessing test passed")
        
        # Test model training (with just one model for speed)
        training_results = pipeline.run_model_training(preprocessing_results, ["custom_cnn"])
        assert training_results is not None
        logger.info("✓ Model training test passed")
        
        # Test evaluation
        evaluation_results = pipeline.run_model_evaluation(training_results, preprocessing_results)
        assert evaluation_results is not None
        logger.info("✓ Model evaluation test passed")
        
        # Test visualization
        visualization_results = pipeline.run_visualization(evaluation_results, training_results)
        assert visualization_results is not None
        logger.info("✓ Visualization test passed")
        
        # Test final report
        final_report = pipeline.generate_final_report(evaluation_results, visualization_results)
        assert final_report.exists()
        logger.info("✓ Final report test passed")
        
        logger.info("🎉 All pipeline tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {str(e)}")
        return False


def test_individual_modules():
    """Test individual modules separately."""
    logger.info("Starting individual module tests")
    
    try:
        # Test data ingestion
        from src.data.ingestion import DataIngestor
        ingester = DataIngestor()
        datasets = ingester.list_available_datasets()
        assert isinstance(datasets, dict)
        logger.info("✓ Data ingestion module test passed")
        
        # Test model factory
        from src.models.architectures import TumorClassificationModels
        factory = TumorClassificationModels()
        model = factory.create_custom_cnn()
        assert model is not None
        logger.info("✓ Model architectures module test passed")
        
        # Test evaluator
        from src.evaluation.metrics import ModelEvaluator
        evaluator = ModelEvaluator(['benign', 'malignant'])
        assert evaluator is not None
        logger.info("✓ Evaluation module test passed")
        
        # Test visualizer
        from src.visualization.plotting import TumorClassificationVisualizer
        visualizer = TumorClassificationVisualizer(['benign', 'malignant'])
        assert visualizer is not None
        logger.info("✓ Visualization module test passed")
        
        logger.info("🎉 All individual module tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Individual module test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting tumor classification pipeline tests")
    
    # Test individual modules first
    modules_passed = test_individual_modules()
    
    if modules_passed:
        # Test complete pipeline
        pipeline_passed = test_pipeline_basic()
        
        if pipeline_passed:
            logger.info("🎉 ALL TESTS PASSED! Pipeline is ready for use.")
        else:
            logger.error("❌ Pipeline test failed")
            sys.exit(1)
    else:
        logger.error("❌ Module tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()