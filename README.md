# Fuzzy AMD - Call Transcript Machine Detection

## Transcripts Data
📁 **Access all call transcripts**: [Google Drive Folder](https://drive.google.com/drive/folders/1ZTIIWky4Qski1lkRmRdIImsZ7DtMR-xX?usp=drive_link)

## Overview

Automated system to analyze call transcripts and identify machine-answered calls using fuzzy pattern matching. Designed for ElevateNow calls under 30 seconds.

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

## Machine Learning Model Training

### BERT-Tiny AMD Classifier

The system includes a machine learning approach using `model_training.py` for enhanced Answering Machine Detection:

#### Model Architecture
- **BERT-Tiny**: Lightweight BERT model (2 layers, 128 hidden size, 2 attention heads)
- **Total Parameters**: ~4.5M (much smaller than full BERT)
- **Input**: User transcript excerpts (max 128 tokens)
- **Output**: Binary classification (Machine/Human) with confidence scores

#### Training Pipeline
```python
# Initialize and train the model
python model_training.py

# Key components:
- TranscriptDataset: Custom PyTorch dataset for transcript classification
- BertTinyAMDClassifier: Neural network with BERT embeddings + classification head
- AMDDataProcessor: Data preparation from fuzzy AMD results
- AMDTrainer: Training loop with validation and early stopping
```

#### Features
- **Automatic Data Preparation**: Converts fuzzy AMD results to training data
- **Cross-Validation**: 80/20 train-validation split
- **Early Stopping**: Prevents overfitting with best model checkpointing
- **Performance Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Visualization**: Training curves and confusion matrix plots
- **Inference**: Single transcript prediction with confidence scores

#### Model Outputs
- `best_bert_tiny_amd.pth` - Best model during training
- `bert_tiny_amd_final.pth` - Final model with metadata
- `bert_tiny_validation_results.csv` - Validation predictions
- `bert_tiny_confusion_matrix.png` - Confusion matrix visualization
- `bert_tiny_training_curves.png` - Training/validation curves

#### Usage Example
```python
from model_training import predict_single_transcript
from transformers import BertTokenizer

# Load tokenizer and make prediction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
is_machine, probability, user_text = predict_single_transcript(
    'bert_tiny_amd_final.pth', 
    tokenizer, 
    transcript_data
)
```

## Key Features

✅ **Robust transcription error handling** ("tone" ↔ "turn")  
✅ **Multiple spelling variations** ("voicemail" ↔ "voice mail")  
✅ **Partial phrase matching** for incomplete transcripts  
✅ **Multi-language support** (English + Hindi patterns)  
✅ **Real-time filtering pipeline** with clear statistics  
✅ **Machine Learning enhancement** with BERT-Tiny classifier  
✅ **Comprehensive training pipeline** with validation and visualization  

## File Structure

```
├── filter.py                    # Main analysis script
├── dict.json                    # Detection patterns dictionary
├── transcripts/                 # Call transcript directories
├── amd_fuzzy_results.csv        # Analysis results (3,549 calls)
├── machine_detected_call_ids.txt # Machine call IDs list
├── model_training.py            # BERT-Tiny training pipeline
└── README.md                    # This documentation
```

This system provides accurate machine detection with robust handling of real-world transcription challenges, enhanced by machine learning capabilities for improved accuracy and scalability.