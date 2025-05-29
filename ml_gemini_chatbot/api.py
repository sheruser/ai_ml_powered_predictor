import google.generativeai as genai

def configure_gemini(api_key):
    """Configure the Gemini API with the provided key."""
    genai.configure(api_key=api_key)

def ask_gemini(prediction, user_data):
    """Generate an explanation using Gemini API."""
    try:
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 300,
            }
        )

        prompt = f"""
        The model predicted the person is {"an Extrovert" if prediction == 1 else "an Introvert"}.
        
        Based on these values:
        - Time Spent Alone: {user_data['Time_spent_Alone']}
        - Stage Fear: {user_data['Stage_fear']}
        - Social Event Attendance: {user_data['Social_event_attendance']}
        - Going Outside: {user_data['Going_outside']}
        - Drained After Socializing: {user_data['Drained_after_socializing']}
        - Friends Circle Size: {user_data['Friends_circle_size']}
        - Post Frequency: {user_data['Post_frequency']}
        
        Please explain why the model might have made this prediction in simple terms for a general audience.
        """

        response = gemini_model.generate_content(prompt)
        return response.text if hasattr(response, 'text') else "No response from Gemini."

    except Exception as e:
        return f"Gemini API Error: {str(e)}"