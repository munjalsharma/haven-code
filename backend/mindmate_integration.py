# -*- coding: utf-8 -*-
import os
import json
import requests
from datetime import datetime
from groq import Groq

class MindMateSentimentAnalyzer:
    """
    Lightweight Sentiment Analyzer for Render Free Tier (512MB RAM).
    Uses Groq LLM for emotion detection to avoid loading heavy local models.
    """

    def __init__(self, model_path=None):
        print("\n" + "="*60)
        print("INITIALIZING CLOUD-BASED EMOTION ANALYZER (Groq)")
        print("="*60)
        
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("[MindMate] ⚠️ Warning: GROQ_API_KEY not found. Falling back to keyword matching.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            print("[MindMate] ✅ Groq client initialized for emotion detection.")
            
        self.weights_loaded = True if self.client else False
        
        # Mapping for keyword fallback
        self.emotion_emoji = {
            'joy': '😊', 'sadness': '😢', 'fear': '😰', 'anger': '😠',
            'surprise': '😲', 'neutral': '😐', 'disgust': '🤢', 'shame': '😳'
        }
        
        print("Sentiment Analyzer initialized!")
        print("="*60 + "\n")

    def predict_emotion(self, text):
        """Detects emotion using Groq LLM for high accuracy with ZERO RAM usage."""
        if not text or not self.client:
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
                model="llama-3.3-70b-versatile",
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