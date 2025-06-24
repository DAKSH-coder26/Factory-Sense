import google.generativeai as genai
import os
from dotenv import load_dotenv

class LLMReasoner:
    def __init__(self, model_name="models/gemini-1.5-flash"):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please create a .env file.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def explain_anomalies(self, anomaly_list):
        responses = []
        for anomaly in anomaly_list:
            prompt = self._build_prompt(anomaly)
            try:
                response = self.model.generate_content(prompt)
                clean_response = response.text.strip().replace('*', '')
                responses.append(clean_response)
                
            except Exception as e:
                responses.append(f"⚠️ Gemini API error: {e}")
        return responses

    def _build_prompt(self, anomaly):
        bottle_id = anomaly['bottle_id']
        anomaly_type = anomaly['type']
        detail = anomaly.get('details', '')
        pos = anomaly.get('position', None)
        expected = anomaly.get('expected_range', None)

        context = f"Context: Anomaly detected for Bottle ID {bottle_id} is '{anomaly_type}'. "
        if expected and pos is not None:
            context += f"It was found at position {pos} but was expected in range {expected}. "
        if detail:
            context += f"Recent state history was {str(detail)[-50:]}. "

        instruction = (
            "Task: You are a factory AI. Respond in exactly two lines.\n"
            "Line 1 must start with 'Problem:' and state the issue.\n"
            "Line 2 must start with 'Action:' and state the immediate corrective action.\n"
            "Example:\n"
            "Problem: Stuck bottle\n"
            "Action: Check conveyor belt for obstructions"
        )

        return context + instruction