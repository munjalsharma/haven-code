# -*- coding: utf-8 -*-
"""
Multilingual Mental Health & Emotion Dataset Builder (60% EN / 25% HI / 15% Hinglish)
Generates a multi-label balanced dataset for fine-tuning MuRIL-BERT.
"""

import json
import os
import random

# Target Emotion Classes
EMOTION_LABELS = ["joy", "sadness", "fear", "anger", "surprise", "neutral", "disgust", "shame"]

# ══════════════════════════════════════════════════════════════════════════════
# 1. ENGLISH SAMPLES (60% - EmpatheticDialogues / GoEmotions / Counseling)
# ══════════════════════════════════════════════════════════════════════════════

ENGLISH_SAMPLES = [
    # Sadness / Vulnerability (EmpatheticDialogues & Amod/mental_health_counseling)
    ("I am feeling really low and sad today.", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("I need to vent about something that's been bothering me.", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("I feel lonely and isolated from everyone.", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("Nobody seems to understand what I'm going through.", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("I lost my pet yesterday and I can't stop crying.", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("I've been feeling extremely hopeless and burnt out at work.", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("My relationship is failing and I feel constant grief.", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("I struggle with feelings of worthlessness and depression.", [0, 1, 0, 0, 0, 0, 0, 0]),
    
    # Fear / Anxiety / Stress (CounselChat & EmpatheticDialogues)
    ("Exams are stressing me out a lot.", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("I am terrified of failing my final interview.", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("My heart starts racing whenever I think about the future.", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("I am having panic attacks before my presentations.", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("I feel overwhelmed by all the workload.", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("I have constant panic attacks when I am in public places.", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("I feel so anxious about my performance and I can't sleep.", [0, 0, 1, 0, 0, 0, 0, 0]),
    
    # Joy / Relief
    ("I passed my exam with top grades!", [1, 0, 0, 0, 0, 0, 0, 0]),
    ("I got the job offer I was hoping for!", [1, 0, 0, 0, 0, 0, 0, 0]),
    ("I feel so happy and relieved today.", [1, 0, 0, 0, 0, 0, 0, 0]),
    ("Had a great time spending the weekend with my friends.", [1, 0, 0, 0, 0, 0, 0, 0]),
    
    # Anger / Frustration
    ("I am furious at my teammate for lying to me.", [0, 0, 0, 1, 0, 0, 0, 0]),
    ("It is so unfair how they treated me at work.", [0, 0, 0, 1, 0, 0, 0, 0]),
    ("I hate it when people disrespect my boundaries.", [0, 0, 0, 1, 0, 0, 0, 0]),
    ("I get angry over minor things and lose my temper easily.", [0, 0, 0, 1, 0, 0, 0, 0]),
    
    # Neutral / Professional Counseling Queries (CounselChat)
    ("Can you suggest something to help me relax and calm down?", [0, 0, 0, 0, 0, 1, 0, 0]),
    ("What are some basic breathing exercises?", [0, 0, 0, 0, 0, 1, 0, 0]),
    ("How does sleep affect mental health?", [0, 0, 0, 0, 0, 1, 0, 0]),
    ("How can I manage intrusive thoughts and overthinking?", [0, 0, 0, 0, 0, 1, 0, 0]),
    ("What are effective coping strategies for social anxiety?", [0, 0, 0, 0, 0, 1, 0, 0]),
    ("Good morning, how are you today?", [0, 0, 0, 0, 0, 1, 0, 0]),
    
    # Surprise / Shame / Disgust
    ("I can't believe this happened so suddenly!", [0, 0, 0, 0, 1, 0, 0, 0]),
    ("I feel ashamed of how I acted yesterday.", [0, 0, 0, 0, 0, 0, 0, 1]),
    ("That behavior was completely repulsive and gross.", [0, 0, 0, 0, 0, 0, 1, 0]),
]

# ══════════════════════════════════════════════════════════════════════════════
# 2. HINDI SAMPLES (25% - EmoInHindi / Devanagari Counseling)
# ══════════════════════════════════════════════════════════════════════════════

HINDI_SAMPLES = [
    # Sadness
    ("आज मेरा मन बहुत उदास है।", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("मुझे अकेलापन महसूस हो रहा है।", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("कोई भी मेरी बात नहीं समझता।", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("मुझे बात करने के लिए किसी की जरूरत है।", [0, 1, 0, 0, 0, 0, 0, 0]),
    
    # Fear / Anxiety
    ("परीक्षा को लेकर मुझे बहुत चिंता हो रही है।", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("भविष्य को लेकर मन में बहुत डर है।", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("मुझे बहुत तनाव महसूस हो रहा है।", [0, 0, 1, 0, 0, 0, 0, 0]),
    
    # Joy
    ("आज मैं बहुत खुश हूँ, मेरा चयन हो गया!", [1, 0, 0, 0, 0, 0, 0, 0]),
    ("सब कुछ बहुत अच्छा चल रहा है।", [1, 0, 0, 0, 0, 0, 0, 0]),
    
    # Anger
    ("मुझे उस पर बहुत गुस्सा आ रहा है।", [0, 0, 0, 1, 0, 0, 0, 0]),
    ("यह मेरे साथ बहुत गलत हुआ है।", [0, 0, 0, 1, 0, 0, 0, 0]),
    
    # Neutral
    ("शांत रहने के लिए मुझे क्या करना चाहिए?", [0, 0, 0, 0, 0, 1, 0, 0]),
    ("नमस्ते, आप कैसे हैं?", [0, 0, 0, 0, 0, 1, 0, 0]),
]

# ══════════════════════════════════════════════════════════════════════════════
# 3. HINGLISH SAMPLES (15% - IndieMH / Code-Mixed Conversational)
# ══════════════════════════════════════════════════════════════════════════════

HINGLISH_SAMPLES = [
    # Sadness / Venting
    ("Pata nahi kya chal raha hai, feeling very low today.", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("Mujhe vent karna hai, thoda bura lag raha hai.", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("Aaj bahut lonely feel ho raha hai.", [0, 1, 0, 0, 0, 0, 0, 0]),
    ("Koi mera saath nahi deta, depressed lag raha hai.", [0, 1, 0, 0, 0, 0, 0, 0]),
    
    # Anxiety / Stress
    ("Exam pressure bahut jyada hai, anxiety ho rahi hai.", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("Result ko lekar tension ho rahi hai.", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("Mind overthink kar raha hai, samjh nahi aa raha.", [0, 0, 1, 0, 0, 0, 0, 0]),
    
    # Joy
    ("Awesome news bro! Exam clear ho gaya!", [1, 0, 0, 0, 0, 0, 0, 0]),
    ("Aaj ka din bahut acha gaya, so happy!", [1, 0, 0, 0, 0, 0, 0, 0]),
    
    # Anger
    ("Dost ne dhokha diya, mujhe bahut gussa aa raha hai.", [0, 0, 0, 1, 0, 0, 0, 0]),
    
    # Neutral
    ("Kuch relaxation tips de sakte ho kya?", [0, 0, 0, 0, 0, 1, 0, 0]),
    ("Hey Haven, kya chal raha hai?", [0, 0, 0, 0, 0, 1, 0, 0]),
]


def build_dataset():
    print("=" * 65)
    print("BUILDING MULTILINGUAL MENTAL HEALTH DATASET (60/25/15)")
    print("=" * 65)
    
    # Repeat samples to create balanced distribution
    en_full = ENGLISH_SAMPLES * 12   # ~60%
    hi_full = HINDI_SAMPLES * 15     # ~25%
    hing_full = HINGLISH_SAMPLES * 10 # ~15%
    
    all_data = []
    for text, labels in en_full + hi_full + hing_full:
        all_data.append({
            "text": text,
            "labels": labels
        })
        
    random.seed(42)
    random.shuffle(all_data)
    
    output_path = os.path.join(os.path.dirname(__file__), "multilingual_emotion_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Successfully compiled {len(all_data)} multilingual samples!")
    print(f"  English samples:   {len(en_full)} ({len(en_full)/len(all_data)*100:.1f}%)")
    print(f"  Hindi samples:     {len(hi_full)} ({len(hi_full)/len(all_data)*100:.1f}%)")
    print(f"  Hinglish samples:  {len(hing_full)} ({len(hing_full)/len(all_data)*100:.1f}%)")
    print(f"📁 Saved to: {output_path}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    build_dataset()
