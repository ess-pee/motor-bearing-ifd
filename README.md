# IFD for Motor Bearing Fault Detection

This repository contains an Intelligent Fault Detection (IFD) System to detect motor bearing faults in industrial machinery. Uses a 1D-CNN to achieve end-to-end classification from raw vibration signal bypassing preprocessing overhead, reducing model size and resulting in fast inference times for deployment on edge devices.

## Overview

In industries like aviation, where undetected bearing faults in aircraft engines can lead to catastrophic failures and millions in damages, rapid and reliable fault detection is critical. Similarly, in manufacturing and energy sectors, bearing failures cause costly downtime, with global industries losing billions annually. By achieving inference times under 100ms and a compact model size, this solution enables real-time monitoring, ensuring safer operations and significant cost savings in high-stakes environments like aircraft maintenance, wind turbines, and industrial machinery.

Traditional fault detection methods rely on computationally intensive preprocessing, such as Fourier or wavelet transforms, which slow down deployment and require expert tuning. These approaches, often paired with manual feature engineering or bulky models like 2D-CNNs, struggle to meet the demands of real-time industrial applications. This 1D-CNN model overcomes these limitations by directly processing raw signals, delivering high accuracy with minimal latency and a lightweight footprint. Designed for scalability, it’s ideal for edge devices in industrial IoT, offering a practical, efficient solution that redefines how bearing faults are detected and managed.

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
│   ├── [dataset]_encoding.json files # Label encodings
|   └── [dataset_samples.npz files # Testing samples
└── playground/         # Development and experimentation files
    ├── training.py     # Main training script
    ├── tf_training.py  # Transfer learning implementation
    ├── tf_evaluation.ipynb # Transfer learning evaluation notebook
    ├── data_preprocessing.py # Data preparation script
    ├── data_extraction_functions.py # Dataset extraction utilities
    ├── utility_functions.py # Common utility functions
    ├── sample_generator.py # Sample data generation
    └── data.7z         # Dataset files (not included in repo) included as a google drive link (check below)
```

## Key Features

- Multiple dataset support (CWRU, MFD, TRI)
  - Case Western Reserve University's Bearing Fault Dataset
  - Federal University of Rio De Janeiro's Machinery Fault Dataset
  - Mehran University of Engineering and Technology's Triaxial Bearing Dataset
- Transfer learning capabilities
- Real-time signal visualization
- RESTful API for predictions
- Production-ready Docker support
- Edge device compatibility
  - Lightweight model architecture
  - Optimized for low-resource environments
  - Minimal memory footprint

## How It Works

This project leverages a meticulously designed 1D Convolutional Neural Network (1D-CNN) to achieve end-to-end classification of motor bearing faults directly from raw vibrational signals, eliminating the need for computationally expensive preprocessing like Fourier or wavelet transforms. The model’s architecture, input processing, and training strategy are optimized for low-latency inference (<100ms on a resource-constrained 0.5 CPU Render server) and a compact footprint, making it a game-changer for real-time fault detection in industries like aviation and manufacturing. By integrating transfer learning, heuristic hyperparameter tuning, and robust evaluation across multiple datasets, this solution demonstrates exceptional performance and adaptability, even in data-scarce industrial scenarios.

### Model Architecture

The 1D-CNN is built with three convolutional blocks, each designed to extract hierarchical features from raw vibrational signals. Each block consists of a 1D convolution layer with a kernel size of 9—chosen to capture long-term time and frequency features, ensuring robustness against short-term noise—followed by ReLU activation and max-pooling (pool size of 2) to reduce dimensionality while preserving critical patterns. The blocks use descending filter sizes (128, 64, 32) to progressively capture low-level features (e.g., raw signal patterns) in the first block and higher-level abstractions (e.g., fault-specific signatures) in later blocks. This is followed by a flatten layer and two dense blocks with 32 and 16 neurons, respectively, each with ReLU activation and dropout (probabilities of 0.4 and 0.2) to mitigate overfitting. The lightweight architecture, combined with strategic downsampling of high-frequency datasets, ensures a compact model size and inference times under 100ms on a 0.5 CPU Render server, ideal for edge deployment.

[Placeholder for Model Architecture Diagram]

### Input and Output

The model processes single-channel raw vibrational signals, requiring only minimal preprocessing: min-max scaling to normalize signal amplitudes and an overlapping sliding window to generate training samples. No additional filtering or feature extraction is applied, enabling true end-to-end learning. For datasets with high sampling rates, such as MAFAULDA’s 50kHz, signals are downsampled to reduce computational load without sacrificing fault detection accuracy, as bearing faults (e.g., cracks) produce distinct frequency signatures even at lower resolutions. The output is a multi-class classification of bearing fault types, which vary by dataset. For example, the CWRU dataset includes four classes: ball fault, inner race fault, outer race fault, and healthy. The MAFAULDA and Triaxial datasets have similar fault categories, though the exact number of classes depends on the dataset’s configuration (typically 4–6 classes). This flexibility allows the model to adapt to diverse industrial motor types.

### Transfer Learning for Real-World Applicability

Collecting faulty motor data in industrial settings is notoriously challenging, as no one risks running a faulty motor to gather data. To address this, the model employs transfer learning to adapt knowledge from a controlled lab environment to real-world scenarios. The model is pretrained on the CWRU Bearing Dataset, a widely used lab-collected dataset, achieving >98% test accuracy. For the target domain, the Triaxial dataset—the noisiest of the three datasets, with the lowest baseline accuracy—was used to simulate real-world conditions. Only 1% of the Triaxial dataset was randomly sampled for fine-tuning, during which the middle two convolutional layers were frozen to retain general feature extraction capabilities, while the first convolutional layer and dense layers were unfrozen. This allowed the model to adapt low-level features to the new domain and refine classification logic, respectively. Over 20 iterations on the same sample (using a fixed random seed), the model achieved an average test accuracy of 77%, a 5.69% improvement over training from scratch on the same 1% data, showcasing robust domain adaptation with minimal data.

### Evaluation and Performance

The model was evaluated on three well-known datasets—CWRU, MAFAULDA, and Triaxial—using an 80-20 train-test split. Across all datasets, it achieved >98% test accuracy when trained on full datasets, demonstrating exceptional robustness. Performance was visualized using confusion matrices to analyze classification accuracy across fault types, revealing consistent performance even on the noisy Triaxial dataset.

[Placeholder for Confusion Matrices]

Compared to four published academic models (including DNNs, 2D-CNNs, and other 1D-CNNs), this model either matches or exceeds performance, outperforming competing 1D-CNNs by approximately 7% in test accuracy while requiring simpler preprocessing. Inference was tested on a 0.5 CPU Render server with limited RAM, confirming <100ms inference times, making it suitable for resource-constrained environments. Training was conducted on a GPU to handle computational demands, but the model’s lightweight design ensures deployment feasibility on low-power devices. Plans for a TensorFlow Lite version are in progress to further optimize inference for edge devices like Raspberry Pi or NVIDIA Jetson.

### Practical Implementation and Challenges

The model’s ability to handle diverse datasets (CWRU, MAFAULDA, Triaxial) demonstrates its robustness across different motor types and operating conditions. While not yet tested in explicitly noisy environments, its performance on the noisy Triaxial dataset suggests potential resilience, with future work planned to validate this. The architecture’s simplicity and downsampling strategy make it deployable on resource-constrained platforms, as evidenced by successful inference on a low-spec Render server. Hyperparameter tuning was largely heuristic, relying on intuition for parameters like learning rate and batch size, which proved effective given the model’s high accuracy. Key challenges included deployment on the resource-limited Render server, which was overcome by optimizing the model’s architecture and downsampling high-frequency data. Documentation remains a time-intensive task, but the model’s smooth development process reflects careful planning and execution.

### Future Enhancements

Future work will focus on testing the model in explicitly noisy environments to further validate robustness, as well as optimizing a TensorFlow Lite version for edge devices like Raspberry Pi or NVIDIA Jetson. Additional evaluation metrics (e.g., precision, recall, F1-score) and robustness tests across varied operating conditions will enhance the model’s industrial applicability. These improvements aim to solidify its potential for real-time fault detection in critical systems like aircraft engines and industrial machinery.

## Playground

The playground directory contains various development and experimentation files that demonstrate the project's evolution and implementation details. These files are crucial for understanding the complete development process of the fault detection system.

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

### Production Server using Docker

1. Build the Docker image:
```bash
docker build -f Dockerfile.app -t ifd-app .
```

2. Run the container:
```bash
docker run -p 8000:8000 ifd-app
```

The application will be available at `http://localhost:8000`, but this is useless to you just go to the [website](https://esspee-ifd.onrender.com/) for the live version.

### Playground

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements/playground_reqs.txt
```

3. Play around with hyperparameters and let me know if you find a better way of doing what I've done

## Production Deployment

The application is configured to run with Gunicorn in production with the following settings:
- 1 worker
- 1 thread per worker
- 90-second timeout
- Maximum 100 requests per worker
- Jitter of 20 requests

## Acknowledgments

[To be added: Credits and acknowledgments] 
