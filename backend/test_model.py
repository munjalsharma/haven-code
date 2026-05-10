import os
import sys
# Add current dir to path for imports
sys.path.append(os.getcwd())

try:
    from mindmate_integration import MindMateSentimentAnalyzer
    print("Initializing analyzer with compressed model...")
    analyzer = MindMateSentimentAnalyzer(model_path="muril_emotion_model.pth")
    
    test_text = "I am feeling very happy today!"
    print(f"Testing prediction on: '{test_text}'")
    result = analyzer.predict_emotion(test_text)
    print(f"Result: {result}")
    
    if result['emotion'] == 'joy' or result['emotion'] == 'neutral':
        print("✅ SUCCESS: Model loaded and predicted correctly.")
    else:
        print(f"❓ WARNING: Predicted {result['emotion']}, check if this is expected.")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
