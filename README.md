# Intelligent Motor Fault Detection (IFD)

A lightweight, edge-compatible web application for intelligent motor fault detection and analysis using optimized machine learning models.

## Overview

This application provides a web interface for motor fault detection using pre-trained machine learning models. It supports multiple datasets and provides visualization capabilities for signal analysis. The models are specifically optimized for edge device deployment, making them suitable for real-time industrial applications.

## Key Features

- Multiple dataset support (CWRU, MFD, TRI)
  - [Dataset descriptions and use cases to be added]
- Transfer learning capabilities
- Real-time signal visualization
- RESTful API for predictions
- Production-ready Docker support
- Edge device compatibility
  - Lightweight model architecture
  - Optimized for low-resource environments
  - Minimal memory footprint

## Project Structure

```
.
├── Dockerfile.app          # Production Docker configuration
├── .dockerignore          # Docker ignore rules
├── requirements/          # Project dependencies
│   ├── production_reqs.txt  # Production environment requirements
│   └── playground_reqs.txt  # Development environment requirements
├── src/                  # Application source code
│   ├── ifd.py           # Main Flask application
│   ├── static/          # Static assets (CSS, JS, images)
│   └── templates/       # HTML templates
│       ├── index.html   # Main landing page
│       ├── cwru.html    # CWRU dataset interface
│       ├── mfd.html     # MFD dataset interface
│       ├── tri.html     # TRI dataset interface
│       └── transfer.html # Transfer learning interface
├── models/              # ML model files
│   ├── cwru_model.h5   # CWRU dataset model
│   ├── mfd_model.h5    # MFD dataset model
│   ├── tri_model.h5    # TRI dataset model
│   ├── transfer_model.h5 # Transfer learning model
│   └── [dataset]_encoding.json files # Label encodings
└── playground/         # Development and experimentation files
    ├── training.py     # Main training script
    ├── tf_training.py  # Alternative TensorFlow implementation
    ├── tf_evaluation.ipynb # Model evaluation notebook
    ├── data_preprocessing.py # Data preprocessing pipeline
    ├── data_extraction_functions.py # Dataset extraction utilities
    ├── utility_functions.py # Common utility functions
    ├── sample_generator.py # Sample data generation
    └── data.7z         # Dataset files (not included in repo)
```

## Edge Device Capabilities

[To be added: Detailed explanation of model optimization techniques and edge device performance metrics]

## Playground

The playground directory contains various development and experimentation files that demonstrate the project's evolution and implementation details. These files are crucial for understanding the complete development process of the fault detection system.

### Development Files

1. **Model Training and Architecture**
   - `training.py`: Main training script that implements the CNN architecture
     - Uses early stopping for efficient training
     - Implements a lightweight CNN architecture optimized for edge devices
     - Supports multiple datasets (CWRU, MFD, TRI)
     - Saves trained models and encoding information

   - `tf_training.py`: Alternative TensorFlow implementation
   - `tf_evaluation.ipynb`: Jupyter notebook for model evaluation and analysis

2. **Data Processing Pipeline**
   - `data_preprocessing.py`: Core preprocessing pipeline
     - Handles data normalization using MinMaxScaler
     - Implements window-based signal segmentation
     - Manages train-test splitting
     - Supports multiple dataset formats

   - `data_extraction_functions.py`: Dataset-specific extraction utilities
     - `extract_cwru_data()`: CWRU dataset extraction
     - `extract_mfd_data()`: MFD dataset extraction
     - `extract_tri_data()`: TRI dataset extraction

3. **Utility Functions**
   - `utility_functions.py`: Common utility functions
     - Signal processing helpers
     - Data transformation utilities
     - Input formatting functions

   - `sample_generator.py`: Sample data generation utilities

4. **Data**
   - `data.7z`: Compressed dataset files (1.1GB)
     - Contains processed and raw data files
     - Note: This file is not included in the repository due to size

### Development Process

The playground demonstrates the complete development workflow:

1. **Data Preparation**
   - Raw signal data extraction
   - Signal segmentation and windowing
   - Normalization and preprocessing
   - Train-test splitting

2. **Model Development**
   - Lightweight CNN architecture design
   - Hyperparameter optimization
   - Training with early stopping
   - Model evaluation and validation

3. **Optimization**
   - Model size optimization for edge deployment
   - Inference time optimization
   - Memory footprint reduction

### Usage Notes

- The playground files are primarily for reference and understanding the development process
- Pre-trained models are already included in the `models/` directory
- The data processing pipeline is optimized for the specific datasets used
- Model architecture is designed for edge device compatibility

## Prerequisites

- Python 3.10.13
- Docker (for containerized deployment)
- TensorFlow/Keras
- Flask

## Quick Start

### Using Docker

1. Build the Docker image:
```bash
docker build -f Dockerfile.app -t ifd-app .
```

2. Run the container:
```bash
docker run -p 8000:8000 ifd-app
```

The application will be available at `http://localhost:8000`

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements/production_reqs.txt
```

3. Run the development server:
```bash
python src/ifd.py
```

## API Endpoints

- `GET /`: Main application page
- `GET /cwru`: CWRU dataset interface
- `GET /mfd`: MFD dataset interface
- `GET /tri`: TRI dataset interface
- `GET /transfer`: Transfer learning interface
- `GET /predict/<dataset>/<sample_number>`: Get prediction for a specific sample

## Model Architecture

[To be added: Detailed explanation of model architecture, optimization techniques, and performance characteristics]

## Production Deployment

The application is configured to run with Gunicorn in production with the following settings:
- 1 worker
- 1 thread per worker
- 90-second timeout
- Maximum 100 requests per worker
- Jitter of 20 requests

## Performance Metrics

[To be added: Model performance metrics, inference times, and resource utilization statistics]

## Live Demo

The application is currently deployed and accessible at: [URL to be added]

## Future Improvements

[To be added: Planned features and improvements]

## License

[To be added]

## Contributing

[To be added]

## Acknowledgments

[To be added: Credits and acknowledgments] 