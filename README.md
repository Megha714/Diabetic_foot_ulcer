# DFU Detection - AI-Powered Diabetic Foot Ulcer Detection

A complete deep learning system for detecting diabetic foot ulcers with built-in validation to reject non-foot images. Uses Vision Transformer architecture with custom computer vision algorithms.

## 🎯 Features

- **Dual Classification**: Detects both Normal vs Abnormal foot conditions
- **Smart Validation**: Rejects non-foot images (faces, hands, objects, etc.)
- **Pre-trained Model**: Ready-to-use model with 98.1% accuracy
- **Web Interface**: User-friendly Flask web application
- **Computer Vision Algorithm**: 7-layer detection system for foot validation
- **Apple Silicon Optimized**: Runs efficiently on M1/M2/M3 Macs using MPS

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Charan1490/Dfu.git
cd Dfu
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and go to: **http://localhost:5000**

That's it! The pre-trained model will load automatically.

## 📊 Model Performance

- **Classification Accuracy**: 98.1%
- **Foot Validation Accuracy**: 100%
- **Training Dataset**: 1,055 foot images + 200 negative samples
- **Architecture**: Vision Transformer (86.9M parameters)

## 🏗️ Project Structure

```
.
├── app.py                          # Flask web application
├── model.py                        # Model architecture
├── foot_detection_algorithm.py    # Computer vision validation
├── enhanced_dataset.py             # Dataset loader with stratification
├── train_with_validation.py        # Training script
├── checkpoints/
│   └── best_model.pth             # Pre-trained model (98.1% acc)
└── templates/
    └── index.html                  # Web UI
```

## 🎓 Training Your Own Model

If you want to retrain with your own data:

1. Organize your dataset:
```
DFU/
├── Patches/
│   ├── Normal(Healthy skin)/
│   └── Abnormal(Ulcer)/
└── Negative_Samples/              # Non-foot images for rejection training
```

2. Run training:
```bash
python train_with_validation.py
```

Training takes ~2 hours on Apple M3 (30 epochs).

## 🔬 How It Works

1. **Computer Vision Pre-screening**: 7 detection methods analyze image features
   - Skin color detection (HSV/YCrCb)
   - Shape analysis (elongation, solidity)
   - Anatomical features (toe detection)
   - Color distribution
   - Texture analysis
   - Edge detection
   - Aspect ratio validation

2. **Neural Network Classification**: Vision Transformer processes validated images
   - Foot validation module (is this a foot?)
   - Classification head (Normal vs Abnormal)

3. **Hybrid Decision**: Combines CV confidence + model confidence for final prediction

## 📝 License

MIT License

## 👤 Author

Charan Naik
