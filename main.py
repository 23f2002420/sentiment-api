from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class CommentRequest(BaseModel):
    comment: str


class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):

    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis tool. "
                        "Analyze the sentiment of the given comment and respond with ONLY a JSON object with exactly two fields:\n"
                        "- sentiment: exactly one of 'positive', 'negative', or 'neutral'\n"
                        "- rating: an integer from 1 (very negative) to 5 (very positive)\n"
                        "Rules: positive = rating 4 or 5, neutral = 3, negative = 1 or 2.\n"
                        "Output only the JSON object, no other text."
                    ),
                },
                {"role": "user", "content": request.comment},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)
        return SentimentResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI API error: {str(e)}")
