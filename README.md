# AMD - Answering Machine Detection System

## 📁 Data Access
**Call Transcripts**: [Google Drive Folder](https://drive.google.com/drive/folders/1ZTIIWky4Qski1lkRmRdIImsZ7DtMR-xX?usp=drive_link)

## 🎯 Overview

Advanced Answering Machine Detection (AMD) system combining rule-based fuzzy pattern matching with state-of-the-art BERT-Tiny machine learning for accurate identification of machine-answered calls. Designed for ElevateNow call center operations with comprehensive training, validation, and production deployment capabilities.

## 🚀 Key Features

✅ **Dual Detection Approach**: Rule-based + Machine Learning  
✅ **BERT-Tiny Model**: Lightweight, efficient transformer architecture  
✅ **Production Ready**: Complete inference pipeline with progressive analysis  
✅ **Real-time Monitoring**: Live training curves and performance tracking  
✅ **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and comparison analysis  
✅ **Robust Error Handling**: Handles transcription errors and edge cases  
✅ **Multi-language Support**: English + Hindi transcript processing  
✅ **Class Imbalance Handling**: Advanced techniques for imbalanced datasets

## Pipeline

```
3,549 Total Calls Analyzed
    ↓ Remove Empty Transcripts
3,548 Non-Empty Calls (99.97%)
    ↓ Remove Single Utterances  
1,020 Multi-Utterance Calls (28.7%)
    ↓ Remove Assistant-Only Calls
753 Calls with User Responses (21.2%)
    ↓ Fuzzy Machine Detection
951 Machine Calls (26.8%) + 2,597 Human Calls (73.2%)
```

## Detection Logic

### Machine Patterns Detected

**Voicemail Messages**
- "voicemail", "voice mail"
- "your call has been forwarded to voicemail"
- "the person you are trying to reach is not available"
- "at the tone", "at the turn" (transcription error)
- "please record your message"
- "when you have finished recording"

**IVR Menu Systems**
- "press 1", "press 2", etc.
- "for sales press", "for support press"
- "please select an option"

**System Messages** 
- "not in service", "invalid number"
- "this number has been disconnected"
- "all circuits are busy now"

**Callback Requests + Numbers**
- "call me back" + phone numbers
- "you can reach me at" + digits

### Matching Strategy

1. **Direct substring match** (100% confidence)
2. **Fuzzy matching** (70%+ similarity for transcription errors)
3. **Word-based matching** (70%+ word overlap for partial phrases)

## Usage

```bash
python3 filter.py
```

**Requirements:**
- Python 3.x
- pandas, difflib
- `transcripts/` folder with call data
- `dict.json` with detection patterns

**Generated Files:**
- `amd_fuzzy_results.csv` - Complete analysis results
- `machine_detected_call_ids.txt` - List of all machine call IDs

## Results Summary

- **Total Processed**: 3,549 calls analyzed
- **Machine Detected**: 951 calls (26.8%)
- **Human Detected**: 2,597 calls (73.2%)
- **Detection Method**: Fuzzy pattern matching with high accuracy
- **Average Call Duration**: ~35 seconds
- **Language Support**: English + Hindi transcripts

## 🤖 Machine Learning Pipeline

### BERT-Tiny AMD Classifier

Advanced machine learning approach using `amd.ipynb` for enhanced Answering Machine Detection with comprehensive training and evaluation:

#### Model Architecture
- **Model**: `prajjwal1/bert-tiny` (2 layers, 128 hidden size, 2 attention heads)
- **Total Parameters**: ~4.4M (lightweight and efficient)
- **Input**: User transcript text (max 128 tokens)
- **Output**: Single logit with sigmoid activation for binary classification
- **Loss Function**: BCEWithLogitsLoss with positive weight for class imbalance

#### Training Features
- **Stratified Data Split**: 80/20 train-validation split maintaining class distribution
- **Class Imbalance Handling**: Positive weight calculation for machine class
- **Advanced Optimizer**: AdamW with weight decay and learning rate scheduling
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Live Monitoring**: Real-time training curves and performance tracking
- **Gradient Clipping**: Prevents exploding gradients during training

#### Comprehensive Evaluation
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Detailed breakdown of predictions
- **Rule-based Comparison**: Side-by-side comparison with fuzzy matching
- **Probability Analysis**: Distribution analysis and threshold optimization
- **Detailed Results**: Individual prediction analysis with confidence scores

#### Generated Files
```
output/
├── best_bert_tiny_amd.pth          # Best model during training
├── bert_tiny_amd_final.pth         # Final model with complete metadata
├── production_inference.py         # Production-ready inference class
├── validation_detailed_results.csv # Individual prediction results
├── bert_tiny_analysis.png          # Comprehensive analysis plots
└── live_training_curves.png        # Training progress visualization
```

#### Production Usage
```python
from output.production_inference import ProductionAMDClassifier

# Initialize classifier
classifier = ProductionAMDClassifier('output/bert_tiny_amd_final.pth')

# Single prediction
is_machine, confidence, user_text = classifier.predict(transcript)

# Progressive analysis (for real-time systems)
utterances = ["Hello?", "Yes, this is John", "Sorry, could you repeat?"]
result = classifier.predict_progressive(utterances)
```

## 📊 Performance Results

### Rule-based System
- **Total Processed**: 3,549 calls analyzed
- **Machine Detected**: 951 calls (26.8%)
- **Human Detected**: 2,597 calls (73.2%)
- **Detection Method**: Fuzzy pattern matching with high accuracy

### BERT-Tiny Model Performance
- **Training Data**: Stratified split maintaining class distribution
- **Validation Accuracy**: High accuracy with detailed metrics
- **Class Imbalance**: Handled with positive weight calculation
- **Agreement Rate**: Comparison with rule-based system
- **Production Ready**: Complete inference pipeline with progressive analysis

## 🛠️ Installation & Usage

### Prerequisites
```bash
# Python 3.12+ required
pip install torch transformers pandas scikit-learn matplotlib seaborn tqdm
```

### Rule-based Detection
```bash
python filter.py
```

### Machine Learning Training
```bash
# Open and run the complete training pipeline
jupyter notebook amd.ipynb
```

### Production Inference
```python
from output.production_inference import ProductionAMDClassifier

# Load trained model
classifier = ProductionAMDClassifier('output/bert_tiny_amd_final.pth')

# Make predictions
is_machine, confidence, text = classifier.predict(transcript)
```

## 📁 Project Structure

```
├── amd.ipynb                     # Complete ML training pipeline
├── filter.py                     # Rule-based detection system
├── dict.json                     # Detection patterns dictionary
├── all_EN_calls.csv              # Call metadata (8,349 calls)
├── transcripts/                  # Call transcript directories
├── output/                       # Generated models and results
│   ├── bert_tiny_amd_final.pth   # Final trained model
│   ├── production_inference.py   # Production inference class
│   ├── validation_detailed_results.csv
│   ├── bert_tiny_analysis.png    # Analysis visualizations
│   └── live_training_curves.png  # Training progress
├── amd_fuzzy_results.csv         # Rule-based analysis results
├── machine_detected_call_ids.txt # Machine call IDs list
└── README.md                     # This documentation
```

## 🎯 Next Steps

1. **Deploy to Production**: Upload model to HuggingFace Hub
2. **Integration**: Connect with VoiceX AMD Manager service
3. **Progressive Analysis**: Implement real-time utterance analysis
4. **Monitoring**: Set up performance tracking on live data
5. **Retraining Pipeline**: Automated model updates with new data

## 📈 Technical Highlights

- **Dual Approach**: Combines rule-based and ML methods for robust detection
- **Lightweight Model**: BERT-Tiny with only 4.4M parameters for efficiency
- **Production Ready**: Complete inference pipeline with error handling
- **Comprehensive Evaluation**: Detailed metrics and comparison analysis
- **Real-time Monitoring**: Live training visualization and progress tracking
- **Class Imbalance**: Advanced techniques for handling skewed datasets

This system provides enterprise-grade machine detection with both rule-based reliability and machine learning sophistication, ready for production deployment in call center operations.