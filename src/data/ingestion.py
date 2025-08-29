"""
Data ingestion module for medical imaging datasets.
Handles downloading and organizing datasets from various sources.
"""

import os
import zipfile
import requests
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Optional imports
try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logging.warning("Kaggle API not available. Kaggle datasets won't be accessible.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple progress fallback
    def tqdm(iterable=None, desc=None, total=None, unit=None, unit_scale=None, unit_divisor=None):
        return iterable or []

from config import RAW_DATA_DIR, DATASET_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


class DataIngestor:
    """Handles data ingestion from various medical imaging sources."""
    
    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_dataset(self, dataset_name: str, force_download: bool = False) -> Path:
        """
        Download dataset from Kaggle.
        
        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
            force_download: Whether to re-download if dataset exists
            
        Returns:
            Path to downloaded dataset directory
        """
        if not KAGGLE_AVAILABLE:
            raise DataIngestionError("Kaggle API not available. Please install kaggle package.")
        
        dataset_path = self.data_dir / dataset_name.split('/')[-1]
        
        if dataset_path.exists() and not force_download:
            logger.info(f"Dataset already exists at {dataset_path}")
            return dataset_path
        
        try:
            logger.info(f"Downloading Kaggle dataset: {dataset_name}")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(self.data_dir), 
                unzip=True
            )
            logger.info(f"Successfully downloaded dataset to {dataset_path}")
            return dataset_path
            
        except Exception as e:
            raise DataIngestionError(f"Failed to download Kaggle dataset {dataset_name}: {str(e)}")
    
    def download_from_url(self, url: str, filename: str, chunk_size: int = 8192) -> Path:
        """
        Download file from URL with progress bar.
        
        Args:
            url: URL to download from
            filename: Local filename to save as
            chunk_size: Download chunk size in bytes
            
        Returns:
            Path to downloaded file
        """
        file_path = self.data_dir / filename
        
        if file_path.exists():
            logger.info(f"File already exists: {file_path}")
            return file_path
        
        try:
            logger.info(f"Downloading from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = file.write(chunk)
                    progress_bar.update(size)
            
            logger.info(f"Successfully downloaded {filename}")
            return file_path
            
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise DataIngestionError(f"Failed to download from {url}: {str(e)}")
    
    def extract_archive(self, archive_path: Path, extract_to: Optional[Path] = None) -> Path:
        """
        Extract archive file.
        
        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to (defaults to same directory as archive)
            
        Returns:
            Path to extracted directory
        """
        if extract_to is None:
            extract_to = archive_path.parent
        
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                logger.info(f"Extracted {archive_path} to {extract_to}")
            else:
                raise DataIngestionError(f"Unsupported archive format: {archive_path.suffix}")
            
            return extract_to
            
        except Exception as e:
            raise DataIngestionError(f"Failed to extract {archive_path}: {str(e)}")
    
    def setup_kaggle_histopathology_dataset(self) -> Dict[str, Any]:
        """
        Setup Kaggle breast histopathology dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if not KAGGLE_AVAILABLE:
            logger.warning("Kaggle API not available. Creating placeholder dataset info.")
            # Return placeholder info for demo
            return {
                'name': 'kaggle_histopathology',
                'path': self.data_dir / "kaggle_placeholder",
                'organized_path': self.data_dir / "kaggle_placeholder",
                'classes': ['0', '1'],
                'description': 'Kaggle dataset (placeholder - requires kaggle package)'
            }
        
        config = DATASET_CONFIGS['kaggle_histopathology']
        dataset_name = config['dataset_name']
        
        try:
            dataset_path = self.download_kaggle_dataset(dataset_name)
            
            # Organize the dataset structure
            organized_path = self.data_dir / "histopathology_organized"
            organized_path.mkdir(exist_ok=True)
            
            dataset_info = {
                'name': 'kaggle_histopathology',
                'path': dataset_path,
                'organized_path': organized_path,
                'classes': config['classes'],
                'description': 'Breast Histopathology Images from Kaggle'
            }
            
            logger.info(f"Kaggle histopathology dataset setup complete: {dataset_info}")
            return dataset_info
            
        except Exception as e:
            raise DataIngestionError(f"Failed to setup Kaggle histopathology dataset: {str(e)}")
    
    def setup_sample_dataset(self) -> Dict[str, Any]:
        """
        Create a sample dataset for testing/demo purposes.
        
        Returns:
            Dictionary with dataset information
        """
        sample_path = self.data_dir / "sample_dataset"
        sample_path.mkdir(exist_ok=True)
        
        # Create sample directory structure
        for class_name in ['benign', 'malignant']:
            class_dir = sample_path / class_name
            class_dir.mkdir(exist_ok=True)
        
        dataset_info = {
            'name': 'sample_dataset',
            'path': sample_path,
            'classes': ['benign', 'malignant'],
            'description': 'Sample dataset for testing pipeline'
        }
        
        logger.info(f"Sample dataset setup complete: {dataset_info}")
        return dataset_info
    
    def list_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available datasets and their configurations.
        
        Returns:
            Dictionary of available datasets
        """
        available = {}
        
        for dataset_name, config in DATASET_CONFIGS.items():
            dataset_path = self.data_dir / dataset_name
            available[dataset_name] = {
                'config': config,
                'exists': dataset_path.exists(),
                'path': dataset_path
            }
        
        # Add sample dataset
        sample_path = self.data_dir / "sample_dataset"
        available['sample_dataset'] = {
            'config': {'classes': ['benign', 'malignant']},
            'exists': sample_path.exists(),
            'path': sample_path
        }
        
        return available


def main():
    """Main function for testing data ingestion."""
    ingester = DataIngestor()
    
    # List available datasets
    datasets = ingester.list_available_datasets()
    print("Available datasets:")
    for name, info in datasets.items():
        print(f"  {name}: {'✓' if info['exists'] else '✗'}")
    
    # Setup sample dataset for testing
    sample_info = ingester.setup_sample_dataset()
    print(f"\nSample dataset created: {sample_info}")


if __name__ == "__main__":
    main()