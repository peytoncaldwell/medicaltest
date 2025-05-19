import openai

# Replace with your own API key here
openai.api_key = "YOUR_API_KEY"

def generate_ai_discharge_notes(chart_content):
    prompt = f"""
    Patient chart:
    {chart_content}

    Write a professional discharge summary based on this chart in plain English.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()