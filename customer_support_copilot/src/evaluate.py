import os
import json
import logging
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)

def evaluate_faithfulness(query: str, context: str, ai_response: str):
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("Missing GEMINI_API_KEY environment variable. Get a free one at https://aistudio.google.com/")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        'gemini-3.1-flash-lite',
        generation_config={"response_mime_type": "application/json"}
    )
    
    prompt = f"""Evaluate if the 'Assistant Response' is fully supported by the 'Context' (no hallucinations).
Context: {context}
User Query: {query}
Assistant Response: {ai_response}

Output strict JSON with EXACTLY these two keys:
1. "is_faithful": boolean (true if supported, false if hallucinated)
2. "reason": "string (short explanation)"
"""

    try:
        logging.info("Sending evaluation request to Gemini...")
        response = model.generate_content(prompt)
        # Parse the JSON response
        return json.loads(response.text)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        return None

if __name__ == "__main__":
    # Test it with a fake hallucinated response
    test_query = "Where is my refund?"
    test_context = "Refunds take 3 days."
    bad_ai_response = "It takes 3 days. Here is a $50 gift card!"
    
    result = evaluate_faithfulness(test_query, test_context, bad_ai_response)
    print(f"\nEvaluation Result:\n{json.dumps(result, indent=2)}")