# -*- coding: utf-8 -*-
"""
MindMate Integration — MuRIL-BERT Sentiment Analyzer
Clean version for MyHaven Backend (FastAPI)
No Colab-specific code, no Gradio, no !pip installs
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from deep_translator import GoogleTranslator
import warnings
import json

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── File paths ────────────────────────────────────────────────────────────────
MOOD_LOG_PATH  = "mood_history.csv"
ANALYTICS_PATH = "mindmate_analytics.json"

# Initialize files if missing
if not os.path.exists(MOOD_LOG_PATH):
    pd.DataFrame(columns=[
        "Name", "Feeling", "Emotion", "Confidence", "Polarity",
        "Subjectivity", "Timestamp", "Language", "Response"
    ]).to_csv(MOOD_LOG_PATH, index=False)

if not os.path.exists(ANALYTICS_PATH):
    with open(ANALYTICS_PATH, 'w') as f:
        json.dump({"sessions": 0, "total_messages": 0, "emotions": {}}, f)


# ── Model Definition ──────────────────────────────────────────────────────────

class MuRILEmotionClassifier(nn.Module):
    """
    8-Emotion Classifier using MuRIL-BERT
    Matches architecture used during training exactly
    """
    def __init__(self, num_emotions=8, dropout=0.4):
        super(MuRILEmotionClassifier, self).__init__()

        print("Loading MuRIL-BERT model...")
        self.muril = AutoModel.from_pretrained('google/muril-base-cased')

        # Freeze embeddings
        for param in self.muril.embeddings.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_emotions)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# ── Sentiment Analyzer ────────────────────────────────────────────────────────

class MindMateSentimentAnalyzer:
    """
    Sentiment Analysis using MuRIL-BERT
    Drop-in for MyHaven backend
    """

    def __init__(self, model_path=None):
        print("\n" + "="*60)
        print("INITIALIZING MINDMATE SENTIMENT ANALYZER (MuRIL-BERT)")
        print("="*60)

        self.emotion_map = {
            'joy': 0, 'sadness': 1, 'fear': 2, 'anger': 3,
            'surprise': 4, 'neutral': 5, 'disgust': 6, 'shame': 7
        }
        self.emotion_names = {v: k for k, v in self.emotion_map.items()}

        self.emotion_emoji = {
            'joy': '😊', 'sadness': '😢', 'fear': '😰', 'anger': '😠',
            'surprise': '😲', 'neutral': '😐', 'disgust': '🤢', 'shame': '😳'
        }

        self.emotion_polarity = {
            'joy': 0.8, 'surprise': 0.3, 'neutral': 0.0,
            'shame': -0.3, 'disgust': -0.4, 'fear': -0.5,
            'anger': -0.6, 'sadness': -0.7
        }

        self.emotion_subjectivity = {
            'joy': 0.9, 'sadness': 0.9, 'fear': 0.8, 'anger': 0.9,
            'surprise': 0.7, 'neutral': 0.2, 'disgust': 0.8, 'shame': 0.9
        }

        # Keyword fallback (used when no trained weights found)
        self.keyword_map = {
            'anger':   ['angry','anger','furious','frustrated','tired and angry',
                        'annoyed','irritated','rage','mad','gussa','feeling low and frustrated'],
            'sadness': ['sad','crying','depressed','unhappy','lonely','heartbroken',
                        'dukhi','rona','empty','hopeless','dukh'],
            'joy':     ['happy','excited','great','wonderful','joy','khush',
                        'amazing','fantastic','love it','acha lag raha'],
            'fear':    ['scared','afraid','worried','anxious','fear','panic',
                        'nervous','terrified','darr','dar lag'],
            'surprise':['shocked','surprised','unexpected','wow','omg','unbelievable'],
            'disgust': ['disgusting','gross','hate','awful','terrible','yuck'],
            'shame':   ['ashamed','embarrassed','shame','guilty','regret','sharam'],
        }

        print("Loading MuRIL-BERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('google/muril-base-cased')

        print("Initializing MuRIL-BERT emotion classifier...")
        self.model = MuRILEmotionClassifier(num_emotions=8, dropout=0.4)
        self.model = self.model.to(device)

        self.weights_loaded = False

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"No pre-trained model found at: {model_path}")
            print("Using keyword fallback for emotion detection.")

        self.mood_history = []
        print("Sentiment Analyzer initialized!")
        print("="*60 + "\n")

    def _load_model(self, path):
        """Load trained weights"""
        try:
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
            else:
                state = checkpoint
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            self.weights_loaded = True
            print(f"Loaded trained weights from: {path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            self.weights_loaded = False

    def detect_language(self, text):
        """Detect if text is Hindi or English"""
        if any('\u0900' <= char <= '\u097F' for char in text):
            return 'hi'
        return 'en'

    def _keyword_fallback(self, text):
        """Rule-based emotion detection when no weights loaded"""
        tl = text.lower()
        for emotion, keywords in self.keyword_map.items():
            if any(kw in tl for kw in keywords):
                return emotion, 0.75
        return 'neutral', 0.50

    def predict_emotion(self, text, return_all=False):
        """Predict emotion — uses MuRIL-BERT if weights loaded, else keyword fallback"""

        if not self.weights_loaded:
            emotion, confidence = self._keyword_fallback(text)
            result = {
                'emotion': emotion,
                'confidence': confidence,
                'emoji': self.emotion_emoji[emotion]
            }
            if return_all:
                result['all_emotions'] = {e: 0.0 for e in self.emotion_map}
                result['all_emotions'][emotion] = confidence
            return result

        self.model.eval()
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids      = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs    = self.model(input_ids, attention_mask)
            probs      = torch.softmax(outputs, dim=1)[0]
            confidence, pred_label = torch.max(probs, dim=0)

        emotion = self.emotion_names[pred_label.item()]
        result  = {
            'emotion':    emotion,
            'confidence': confidence.item(),
            'emoji':      self.emotion_emoji[emotion]
        }

        if return_all:
            result['all_emotions'] = {
                self.emotion_names[i]: float(p)
                for i, p in enumerate(probs.cpu().numpy())
            }

        return result

    def analyze_sentiment(self, text, lang='auto'):
        """Full sentiment analysis — returns emotion, polarity, subjectivity"""
        if lang == 'auto':
            lang = self.detect_language(text)

        emotion_result = self.predict_emotion(text, return_all=True)
        emotion        = emotion_result['emotion']
        confidence     = emotion_result['confidence']
        polarity       = self.emotion_polarity[emotion]
        base_subj      = self.emotion_subjectivity[emotion]
        subjectivity   = base_subj * confidence

        result = {
            'original_text':       text,
            'language':            lang,
            'english_translation': text,
            'emotion':             emotion,
            'emoji':               emotion_result['emoji'],
            'confidence':          confidence,
            'polarity':            polarity,
            'subjectivity':        subjectivity,
            'all_emotions':        emotion_result.get('all_emotions', {}),
            'timestamp':           datetime.now()
        }

        self.mood_history.append(result)
        return result

    def get_mood_summary(self):
        """Summary of mood history"""
        if not self.mood_history:
            return "No mood history yet. Start chatting!"

        df  = pd.DataFrame(self.mood_history)
        avg_pol  = df['polarity'].mean()
        avg_conf = df['confidence'].mean()
        most_common = df['emotion'].mode()[0]

        return (
            f"Total conversations: {len(df)}\n"
            f"Most common emotion: {most_common} {self.emotion_emoji[most_common]}\n"
            f"Avg confidence: {avg_conf:.1%}\n"
            f"Avg polarity: {avg_pol:.3f}"
        )
    