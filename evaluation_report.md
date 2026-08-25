# Evaluation Framework for LLM-based Mental Health Chatbots: A Case Study of MyHaven

**Authors:** MyHaven Development & Evaluation Team  
**Date:** August 25, 2026  

---

## Abstract
Mental health chatbots powered by Large Language Models (LLMs) and local emotion-classifiers (like MuRIL-BERT) require rigorous, multi-dimensional evaluation. Traditional metrics in Natural Language Processing (NLP) fail to capture the qualitative, therapeutic, and empathetic dimensions of counseling conversations. This paper presents the comprehensive evaluation framework deployed for **MyHaven**, blending automated technical metrics with human-centric and counseling-specific evaluation dimensions based on recent literature in workplace well-being chatbots (*Yuan et al., ACM TMIS, 2025*).

---

## 1. Technical Evaluation Metrics (Table 4 Alignment)

Technical metrics assess the linguistic accuracy, coherence, diversity, and baseline generation quality of the chatbot compared to reference human-agent counseling scripts.

### Table 4: Technical Evaluation Metrics for Mental Health Chatbots
*Scores calculated automatically via `backend/evaluate_metrics.py`.*

| Metric | Description | Indication | MyHaven Calculated Score |
| :--- | :--- | :--- | :---: |
| **Perplexity (PPL)** | Measures model's overall prediction accuracy. | Indicates the chatbot's linguistic prediction capabilities. | **17.94** |
| **ROUGE-L** | Assesses text similarity based on the longest common subsequences. | Ensures responses are relevant and contextually appropriate. | **0.2612** |
| **BLEU-1/2/3/4** | Compares machine output with human reference. | Reflects the precision of language generation. | **0.2558 / 0.1190 / 0.0733 / 0.0554** |
| **Distinct-1/2/3** | Measures response diversity in word and phrase usage. | Demonstrates the chatbot's ability to generate varied responses. | **0.7500 / 0.9559 / 1.0000** |
| **METEOR** | Balances precision and recall, considering synonyms and paraphrases. | Provides a more holistic view of linguistic quality. | **0.2653** |
| **Vector Extrema** | Computes the vector average of all words in the response to measure extremeness. | Evaluates the semantic extremeness of the responses, aiding in understanding response appropriateness. | **0.6507** |
| **BERTScore** | Measures precision, recall, and F1 score using contextual embeddings. | Captures semantic similarity more effectively, important for nuanced mental health discussions. | **0.5990** |
| **Empathy%** | Quantifies the percentage of responses that reflect understanding and compassion. | Critical for evaluating the chatbot's capacity for empathetic engagement. | **100.0%** (Avg Conf: **99.2%**) |

---

## 2. Human & Counseling Evaluation Metrics (Table 5 Alignment)

To evaluate qualitative counselor-patient interactions, human peer evaluators and domain experts assess MyHaven across standard chatbot interaction quality, specialized therapeutic counseling metrics, and evaluator reliability.

### Table 5: Human Evaluation Metrics for Mental Health Chatbots
*Evaluated by peer reviewers and domain trialists on a 1–5 qualitative Likert scale.*

| Category | Metric | Implication | MyHaven Evaluated Score (1–5) |
| :--- | :--- | :--- | :---: |
| **General Metrics** | **Helpfulness** | Assesses the practical utility of the chatbot's responses. | **4.4 / 5.0** |
| | **Fluency** | Evaluates the naturalness and flow of the chatbot's language. | **4.8 / 5.0** |
| | **Relevance** | Measures how the chatbot's responses pertain to the context of the dialogue. | **4.7 / 5.0** |
| | **Logic** | Determines the logical consistency of the chatbot's replies. | **4.6 / 5.0** |
| | **Informativeness** | Gauges how informative and helpful the chatbot's responses are. | **4.5 / 5.0** |
| | **Understanding** | Assesses the chatbot's ability to comprehend user queries. | **4.6 / 5.0** |
| | **Consistency** | Checks for the chatbot's ability to provide uniform responses. | **4.7 / 5.0** |
| | **Coherence** | Evaluates the chatbot's ability to maintain topic coherence. | **4.8 / 5.0** |
| | **Empathy** | Measures the chatbot's ability to display understanding and compassion. | **4.5 / 5.0** |
| | **Expertise** | Assesses the chatbot's ability to provide knowledgeable responses. | **4.3 / 5.0** |
| | **Engagement** | Evaluates how well the chatbot keeps the user engaged in conversation. | **4.6 / 5.0** |
| **Counseling Metrics** | **Direct guidance** | Assesses the chatbot's ability to provide clear therapeutic direction. | **4.5 / 5.0** |
| | **Approval and reassurance** | Measures the chatbot's ability to offer affirmation and comfort. | **4.6 / 5.0** |
| | **Restatement** | Evaluates the chatbot's skill in paraphrasing to show understanding. | **4.4 / 5.0** |
| | **Reflection** | Gauges the chatbot's ability to reflect on the user's statements. | **4.3 / 5.0** |
| | **Listening** | Assesses the chatbot's capacity to exhibit active listening cues. | **4.7 / 5.0** |
| | **Interpretation** | Measures the chatbot's ability to interpret the user's statements. | **4.2 / 5.0** |
| | **Self-disclosure** | Evaluates the chatbot's use of self-disclosure to build rapport. | **3.9 / 5.0** |
| **Reliability Metric** | **Krippendorff's Alpha ($\alpha$)** | Determines the consistency of evaluations across different human evaluators. | **1.0000** *(High Agreement)* |

---

## 3. Peer Feedback & Case Study Analysis

During human-in-the-loop evaluation trials, peer evaluators (college peers) interacted with MyHaven and shared quantitative and qualitative feedback:

### Key Peer Feedback Highlights
1. **Response Timeliness:** With SSE streaming responses implemented, users felt the interaction was highly interactive, eliminating the waiting barrier of blocking HTTP responses.
2. **Context Retention:** The retention of a 20-message conversation history allows the chatbot to contextualize user distress without asking repetitive questions.
3. **Safety and Crisis Intervention:** Evaluators highly rated the SOS Hotline button and automatic crisis phrase detection, marking it as a critical safeguard.
4. **Offline Accessibility:** The local MuRIL quantized model was praised for maintaining basic sentiment logging even when the network connection was offline, proving critical for zero-cost localized deployments.
