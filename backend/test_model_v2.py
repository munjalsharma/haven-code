import os
import sys
sys.path.append(os.getcwd())

try:
    from mindmate_integration import MindMateSentimentAnalyzer
    print("Initializing analyzer...")
    analyzer = MindMateSentimentAnalyzer(model_path="muril_emotion_model.pth")
    
    test_text = "I am feeling very happy today!"
    result = analyzer.predict_emotion(test_text)
    # Remove emoji for printing
    safe_result = {k: v for k, v in result.items() if k != 'emoji'}
    print("Prediction successful.")
    print(f"Emotion: {safe_result['emotion']}, Confidence: {safe_result['confidence']:.4f}")
    
    if analyzer.weights_loaded:
        print("SUCCESS: Weights were loaded successfully from the compressed file.")
    else:
        print("FAILURE: Weights were NOT loaded (keyword fallback used).")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
