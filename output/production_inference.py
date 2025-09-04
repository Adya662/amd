
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Tuple

class ProductionAMDClassifier:
    """Production-ready AMD classifier using BERT-Tiny with single logit output"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load model and tokenizer
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']

        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'], num_labels=1  # Single logit output
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.max_length = config['max_length']
        self.threshold = checkpoint.get('threshold', 0.5)

        print(f"Loaded AMD classifier on {self.device}")
        print(f"Architecture: Single logit + BCEWithLogitsLoss")

    @torch.no_grad()
    def predict(self, transcript: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        """Predict if transcript is from answering machine"""

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

        # Tokenize and predict
        encoding = self.tokenizer(
            user_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logit = outputs.logits.squeeze(-1)  # Single logit value
        machine_prob = torch.sigmoid(logit).item()  # Apply sigmoid

        is_machine = machine_prob >= self.threshold

        return is_machine, machine_prob, user_text

    def predict_progressive(self, utterances: List[str], 
                          stage_thresholds: List[float] = [0.95, 0.85, 0.75]) -> Dict[str, Any]:
        """
        Progressive utterance analysis for production AMD system

        Args:
            utterances: List of user utterance texts
            stage_thresholds: Confidence thresholds for each stage

        Returns:
            Dictionary with decision, confidence, stage, and metadata
        """
        results = {
            'final_decision': False,
            'confidence': 0.0,
            'decision_stage': 0,
            'stage_results': [],
            'utterances_processed': 0
        }

        for stage, utterance_count in enumerate([1, 2, 3], 1):
            if len(utterances) < utterance_count:
                break

            # Combine utterances up to current stage
            combined_text = " ".join(utterances[:utterance_count])

            # Get prediction
            is_machine, confidence, _ = self.predict([
                {"speaker": "user", "content": combined_text}
            ])

            stage_result = {
                'stage': stage,
                'utterances': utterance_count,
                'confidence': confidence,
                'text': combined_text[:100] + '...'
            }
            results['stage_results'].append(stage_result)
            results['utterances_processed'] = utterance_count

            # Check if confidence meets threshold for this stage
            if stage <= len(stage_thresholds) and confidence >= stage_thresholds[stage-1]:
                results['final_decision'] = is_machine
                results['confidence'] = confidence
                results['decision_stage'] = stage
                break

            # Final stage - make decision regardless of confidence
            if stage == 3:
                results['final_decision'] = is_machine
                results['confidence'] = confidence
                results['decision_stage'] = stage

        return results

# Usage example:
# classifier = ProductionAMDClassifier('output/bert_tiny_amd_final.pth')
# 
# # Single prediction
# is_machine, confidence, user_text = classifier.predict(transcript)
# 
# # Progressive analysis (for production system)
# utterances = ["Hello?", "Yes, this is John", "Sorry, could you repeat that?"]
# result = classifier.predict_progressive(utterances)
