# -*- coding: utf-8 -*-
import os
import json
import requests
from datetime import datetime
from groq import Groq

# Dynamic imports check for PyTorch and Transformers to support Render Free Tier compatibility
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

MURIL_BASE = "google/muril-base-cased"
EMOTION_LABELS = ["joy", "sadness", "fear", "anger", "surprise", "neutral", "disgust", "shame"]
EMOJI_MAP = {
    'joy': '😊', 'sadness': '😢', 'fear': '😰', 'anger': '😠',
    'surprise': '😲', 'neutral': '😐', 'disgust': '🤢', 'shame': '😳'
}

if HAS_TORCH:
    class MuRILEmotionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.muril = AutoModel.from_pretrained(MURIL_BASE)
            # Freeze embeddings layer parameters to save resources/training memory
            for p in self.muril.embeddings.parameters():
                p.requires_grad = False
            self.drop = nn.Dropout(0.4)
            self.classifier = nn.Sequential(
                nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.4), nn.BatchNorm1d(512),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.BatchNorm1d(256),
                nn.Linear(256, len(EMOTION_LABELS))
            )

        def forward(self, ids, mask):
            o = self.muril(input_ids=ids, attention_mask=mask, return_dict=True)
            return self.classifier(self.drop(o.last_hidden_state[:, 0]))
else:
    class MuRILEmotionClassifier:
        pass


class MindMateSentimentAnalyzer:
    """
    Lightweight Sentiment Analyzer for Render Free Tier (512MB RAM).
    Uses Groq LLM for emotion detection to avoid loading heavy local models.
    """

    def __init__(self, model_path=None):
        print("\n" + "="*60)
        print("INITIALIZING EMOTION ANALYZER")
        print("="*60)
        
        self.weights_loaded = False
        self.client = None
        self.device = None
        self.model = None
        self.tokenizer = None
        
        # Mapping for keyword fallback
        self.emotion_emoji = EMOJI_MAP
        
        if HAS_TORCH:
            # Determine weights file path
            here = os.path.dirname(os.path.abspath(__file__))
            paths_to_check = []
            if model_path:
                paths_to_check.append(model_path)
                paths_to_check.append(os.path.join(here, model_path))
            
            env_weights = os.getenv("MURIL_EMOTION_WEIGHTS", "")
            if env_weights:
                paths_to_check.append(env_weights)
                
            paths_to_check.extend([
                os.path.join(here, "models", "muril_emotion_model.pth"),
                os.path.join(here, "muril_emotion_model.pth"),
            ])
            
            found_weights_path = None
            for path in paths_to_check:
                if path and os.path.exists(path):
                    found_weights_path = path
                    break
            
            if found_weights_path:
                try:
                    print(f"[MindMate] Loading local weights from: {found_weights_path}")
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"[MindMate] Using device: {self.device}")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(MURIL_BASE)
                    self.model = MuRILEmotionClassifier().to(self.device)
                    self.model.eval()
                    
                    checkpoint = torch.load(found_weights_path, map_location=self.device)
                    if isinstance(checkpoint, dict):
                        if "model_state_dict" in checkpoint:
                            state_dict = checkpoint["model_state_dict"]
                        elif "state_dict" in checkpoint:
                            state_dict = checkpoint["state_dict"]
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    self.model.load_state_dict(state_dict, strict=False)
                    self.weights_loaded = True
                    print("[MindMate] SUCCESS: MuRIL local model loaded successfully.")
                except Exception as e:
                    print(f"[MindMate] ERROR: Error loading local MuRIL model: {e}")
                    self.weights_loaded = False
            else:
                print("[MindMate] WARNING: Local weights file not found.")
        else:
            print("[MindMate] INFO: PyTorch or Transformers not available. Local model disabled.")
            
        if not self.weights_loaded:
            print("[MindMate] INITIALIZING CLOUD-BASED EMOTION ANALYZER (Groq fallback)")
            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                print("[MindMate] WARNING: GROQ_API_KEY not found. Falling back to keyword matching.")
                self.client = None
            else:
                self.client = Groq(api_key=self.api_key)
                print("[MindMate] SUCCESS: Groq client initialized for emotion detection.")
                
        print("Sentiment Analyzer initialized!")
        print("="*60 + "\n")

    def predict_emotion(self, text, previous_context=None, temperature=1.8):
        """Detects emotion using local MuRIL model with Temperature Scaling, Sigmoid Multi-Label scoring, and Multi-Turn Context."""
        if not text:
            return self._keyword_fallback(text)
            
        if self.weights_loaded:
            try:
                # 1. Multi-Turn Context Concatenation: (previous_turn + " [SEP] " + current_text)
                full_text = text
                if previous_context:
                    full_text = f"{previous_context} [SEP] {text}"

                inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=128)
                ids = inputs["input_ids"].to(self.device)
                mask = inputs["attention_mask"].to(self.device)
                
                with torch.no_grad():
                    logits = self.model(ids, mask)
                    
                    # 2. Temperature Scaling & Calibration (z / T)
                    scaled_logits = logits / temperature
                    
                    # 3. Multi-Label Sigmoid Probabilities (allowing overlapping mixed emotions)
                    sigmoid_probs = torch.sigmoid(scaled_logits).squeeze(0)
                    softmax_probs = torch.softmax(scaled_logits, dim=1).squeeze(0)
                    
                confidence, predicted_idx = torch.max(softmax_probs, dim=0)
                confidence = float(confidence.item())
                predicted_idx = int(predicted_idx.item())
                
                emotion = EMOTION_LABELS[predicted_idx]
                emoji = EMOJI_MAP.get(emotion, "😐")

                # Build multi-label breakdown dictionary
                multi_label_scores = {
                    EMOTION_LABELS[i]: round(float(sigmoid_probs[i].item()), 4)
                    for i in range(len(EMOTION_LABELS))
                }

                # Post-processing smoothing for meta/introductory statements (e.g., "I need to vent...")
                text_lower = text.lower()
                if emotion == "anger":
                    meta_vent_phrases = ["need to vent", "want to vent", "bothering me", "troubling me", "can i talk", "need to talk", "listen to me"]
                    aggressive_words = ["hate", "stupid", "idiot", "shut up", "kill", "rage", "furious", "mad at", "curse", "ugly"]
                    if any(p in text_lower for p in meta_vent_phrases) and not any(w in text_lower for w in aggressive_words):
                        emotion = "sadness"
                        emoji = EMOJI_MAP.get(emotion, "😢")
                        confidence = 0.88

                return {
                    "emotion": emotion,
                    "confidence": confidence,
                    "emoji": emoji,
                    "multi_label_scores": multi_label_scores
                }
            except Exception as e:
                print(f"[MindMate] Local MuRIL Inference Error: {e}")
                # Fall back to Groq if possible, or keyword fallback
                if not self.client:
                    return self._keyword_fallback(text)
                    
        # If local weights not loaded or failed, fall back to Groq
        if not self.client:
            return self._keyword_fallback(text)
            
        try:
            # Use a fast model for emotion detection
            prompt = f"""
            Analyze the emotion of the following text: "{text}"
            Return ONLY a JSON object with these fields:
            - emotion: (one of: joy, sadness, anger, fear, surprise, neutral, disgust, shame)
            - confidence: (a float between 0.9 and 0.99)
            - emoji: (a single matching emoji)
            
            Example: {{"emotion": "joy", "confidence": 0.98, "emoji": "😊"}}
            """
            
            completion = self.client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "groq/compound-mini").strip(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(completion.choices[0].message.content)
            # Ensure fields exist
            if 'emotion' not in result: result['emotion'] = 'neutral'
            if 'emoji' not in result: result['emoji'] = self.emotion_emoji.get(result['emotion'], '😐')
            if 'confidence' not in result: result['confidence'] = 0.95
            
            return result

        except Exception as e:
            print(f"[MindMate] Groq Emotion Error: {e}")
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text):
        """Simple rule-based fallback if API fails or is missing."""
        text = text.lower()
        if any(word in text for word in ['happy', 'good', 'great', 'joy', 'excited', 'khush']):
            return {"emotion": "joy", "confidence": 0.85, "emoji": "😊"}
        if any(word in text for word in ['sad', 'bad', 'depressed', 'cry', 'sorry', 'dukhi']):
            return {"emotion": "sadness", "confidence": 0.85, "emoji": "😢"}
        if any(word in text for word in ['angry', 'mad', 'hate', 'annoyed', 'gussa']):
            return {"emotion": "anger", "confidence": 0.85, "emoji": "😠"}
        if any(word in text for word in ['scared', 'afraid', 'fear', 'darr']):
            return {"emotion": "fear", "confidence": 0.85, "emoji": "😰"}
        return {"emotion": "neutral", "confidence": 0.70, "emoji": "😐"}

    def analyze_sentiment(self, text, lang='auto'):
        """Full sentiment analysis compatibility wrapper for main.py."""
        res = self.predict_emotion(text)
        
        return {
            'original_text': text,
            'language': 'auto',
            'english_translation': text,
            'emotion': res['emotion'],
            'emoji': res['emoji'],
            'confidence': res['confidence'],
            'polarity': 0.0, # Placeholder
            'subjectivity': 0.0, # Placeholder
            'timestamp': datetime.now()
        }

    def get_mood_summary(self):
        """Mock mood summary for compatibility."""
        return "Mood tracking is active. History is saved in the database."