import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TranscriptDataset(Dataset):
    """Dataset class for transcript classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class BertTinyAMDClassifier(nn.Module):
    """BERT-Tiny based binary classifier for Answering Machine Detection"""
    
    def __init__(self, dropout_rate=0.3):
        super(BertTinyAMDClassifier, self).__init__()
        
        # Load BERT-Tiny configuration
        # BERT-Tiny: 2 layers, 128 hidden size, 2 attention heads
        self.bert_config = BertConfig(
            vocab_size=30522,  # Standard BERT vocab
            hidden_size=128,   # Tiny model
            num_hidden_layers=2,  # Tiny model
            num_attention_heads=2,  # Tiny model
            intermediate_size=512,  # 4x hidden_size
            max_position_embeddings=512,
            type_vocab_size=2,
            pad_token_id=0
        )
        
        # Initialize BERT model with tiny config
        self.bert = BertModel(self.bert_config)
        
        # Classification head - single neuron for binary classification
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert_config.hidden_size, 1)  # 128 -> 1
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class AMDDataProcessor:
    """Data processor to prepare training data from transcripts"""
    
    def __init__(self, fuzzy_amd_instance):
        self.fuzzy_amd = fuzzy_amd_instance
    
    def extract_user_text(self, transcript):
        """Extract and combine all user utterances from transcript"""
        if not transcript:
            return ""
        
        user_texts = []
        for utterance in transcript:
            if utterance.get("speaker", "").lower() == "user":
                content = utterance.get("content", "").strip()
                if content:
                    user_texts.append(content)
        
        return " ".join(user_texts)
    
    def prepare_training_data(self, call_data, test_size=0.2, random_state=42):
        """
        Prepare training data from call data using existing rule-based labels
        
        Args:
            call_data: List of (call_id, transcript_url, duration) tuples
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            train_texts, val_texts, train_labels, val_labels
        """
        print("Preparing training data...")
        
        texts = []
        labels = []
        call_ids = []
        
        # Process each call
        for call_id, transcript_url, duration in tqdm(call_data, desc="Processing transcripts"):
            # Load transcript
            transcript = self.fuzzy_amd._load_transcript(call_id)
            if not transcript:
                continue
            
            # Extract user text
            user_text = self.extract_user_text(transcript)
            if not user_text.strip():
                continue
            
            # Get label from rule-based detection
            is_machine, reason = self.fuzzy_amd._detect_machine(user_text)
            
            texts.append(user_text)
            labels.append(1 if is_machine else 0)  # 1 for machine, 0 for human
            call_ids.append(call_id)
        
        print(f"Processed {len(texts)} valid transcripts")
        print(f"Machine calls: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        print(f"Human calls: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")
        
        # Split into train/validation
        train_texts, val_texts, train_labels, val_labels, train_ids, val_ids = train_test_split(
            texts, labels, call_ids, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        return train_texts, val_texts, train_labels, val_labels, train_ids, val_ids

class AMDTrainer:
    """Training class for BERT-Tiny AMD classifier"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = criterion(logits.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits.squeeze(), labels)
                
                total_loss += loss.item()
                
                # Convert logits to predictions using sigmoid
                probs = torch.sigmoid(logits.squeeze())
                preds = (probs > 0.5).float()
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy, predictions, true_labels
    
    def train(self, train_dataloader, val_dataloader, epochs=10, lr=2e-5):
        """Full training loop"""
        criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(val_dataloader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_bert_tiny_amd.pth')
                print(f"New best model saved! Accuracy: {val_acc:.4f}")
    
    def evaluate(self, dataloader):
        """Detailed evaluation with metrics"""
        self.model.eval()
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(logits.squeeze())
                preds = (probs > 0.5).float()
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        print("\n=== EVALUATION RESULTS ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Human', 'Machine'], 
                   yticklabels=['Human', 'Machine'])
        plt.title('Confusion Matrix - BERT-Tiny AMD Classifier')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('bert_tiny_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels
        }
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('bert_tiny_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training pipeline"""
    
    # 1. SETUP
    print("=== BERT-Tiny AMD Classifier Setup ===")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer (using standard BERT tokenizer)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
    
    # 2. DATA PREPARATION
    print("\n=== Data Preparation ===")
    
    # Initialize your existing FuzzyAMD system to generate labels
    from filter import FuzzyAMD, load_call_data_from_csv  # Import your existing code
    
    # Load your call data
    CSV_FILE = "all_EN_calls.csv"  # Update path as needed
    call_data = load_call_data_from_csv(CSV_FILE)
    
    # Initialize FuzzyAMD for rule-based labeling
    fuzzy_amd = FuzzyAMD(
        s3_bucket="voicex-call-recordings",
        dict_path="dict.json",
        fuzzy_threshold=0.9
    )
    
    # Process data
    data_processor = AMDDataProcessor(fuzzy_amd)
    train_texts, val_texts, train_labels, val_labels, train_ids, val_ids = \
        data_processor.prepare_training_data(call_data, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # 3. CREATE DATASETS AND DATALOADERS
    print("\n=== Creating Data Loaders ===")
    
    # Create datasets
    train_dataset = TranscriptDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = TranscriptDataset(val_texts, val_labels, tokenizer, max_length=128)
    
    # Create data loaders
    batch_size = 16  # Small batch size for BERT-Tiny
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    
    # 4. MODEL INITIALIZATION
    print("\n=== Model Initialization ===")
    
    # Create model
    model = BertTinyAMDClassifier(dropout_rate=0.3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model architecture:")
    print(f"  - BERT layers: {model.bert_config.num_hidden_layers}")
    print(f"  - Hidden size: {model.bert_config.hidden_size}")
    print(f"  - Attention heads: {model.bert_config.num_attention_heads}")
    print(f"  - Classification head: {model.bert_config.hidden_size} -> 1")
    
    # 5. TRAINING
    print("\n=== Training ===")
    
    trainer = AMDTrainer(model, device)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=10,  # Start with 10 epochs
        lr=2e-5     # Learning rate for BERT-Tiny
    )
    
    # 6. EVALUATION
    print("\n=== Final Evaluation ===")
    
    # Load best model
    model.load_state_dict(torch.load('best_bert_tiny_amd.pth'))
    trainer.model = model
    
    # Evaluate on validation set
    results = trainer.evaluate(val_dataloader)
    
    # Plot training curves
    trainer.plot_training_history()
    
    # 7. SAVE MODEL AND RESULTS
    print("\n=== Saving Results ===")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.bert_config,
        'tokenizer_vocab': tokenizer.vocab_size,
        'max_length': 128,
        'results': results
    }, 'bert_tiny_amd_final.pth')
    
    # Save predictions for analysis
    val_results_df = pd.DataFrame({
        'call_id': val_ids,
        'true_label': results['true_labels'],
        'predicted_label': results['predictions'],
        'probability': results['probabilities'],
        'text_sample': [text[:100] + '...' for text in val_texts]
    })
    val_results_df.to_csv('bert_tiny_validation_results.csv', index=False)
    
    print("Training completed!")
    print("Files saved:")
    print("  - best_bert_tiny_amd.pth (best model during training)")
    print("  - bert_tiny_amd_final.pth (final model + metadata)")
    print("  - bert_tiny_validation_results.csv (validation predictions)")
    print("  - bert_tiny_confusion_matrix.png")
    print("  - bert_tiny_training_curves.png")
    
    return model, trainer, results

def predict_single_transcript(model_path, tokenizer, transcript, device='cpu'):
    """
    Function to make prediction on a single transcript
    
    Args:
        model_path: Path to saved model
        tokenizer: BERT tokenizer
        transcript: Transcript in the format you showed
        device: Device to run inference on
    
    Returns:
        is_machine: Boolean prediction
        probability: Float probability of being machine
        user_text: Extracted user text used for prediction
    """
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model
    model = BertTinyAMDClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Extract user text
    user_texts = []
    for utterance in transcript:
        if utterance.get("speaker", "").lower() == "user":
            content = utterance.get("content", "").strip()
            if content:
                user_texts.append(content)
    
    user_text = " ".join(user_texts)
    
    if not user_text.strip():
        return False, 0.0, ""
    
    # Tokenize
    encoding = tokenizer(
        user_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        logits = model(input_ids, attention_mask)
        probability = torch.sigmoid(logits).item()
        is_machine = probability > 0.5
    
    return is_machine, probability, user_text

# Example usage
if __name__ == "__main__":
    # Run the main training pipeline
    model, trainer, results = main()
    
    # Example inference on new transcript
    example_transcript = [
        {"timestamp": "2025-08-21 11:38:19", "speaker": "assistant", "content": "Hello, this is Priya from ElevateNow. Am I speaking with Pooja sahni Sahni?"},
        {"timestamp": "2025-08-21 11:38:21", "speaker": "user", "content": "Yes?"},
        {"timestamp": "2025-08-21 11:38:21", "speaker": "assistant", "content": "Great! I noticed you were checking out our Nano plan but didn't complete the payment. I'd love to understand what motivated you to consider"}
    ]
    
    # Load tokenizer for inference
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Make prediction
    is_machine, prob, user_text = predict_single_transcript(
        'bert_tiny_amd_final.pth', 
        tokenizer, 
        example_transcript
    )
    
    print(f"\n=== EXAMPLE PREDICTION ===")
    print(f"User text: '{user_text}'")
    print(f"Prediction: {'Machine' if is_machine else 'Human'}")
    print(f"Confidence: {prob:.4f}")