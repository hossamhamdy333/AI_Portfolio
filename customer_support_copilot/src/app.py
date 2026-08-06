from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI Support Copilot API")

class ChatRequest(BaseModel):
    query: str
    context: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
    
        if "refund" in request.query.lower():
            ai_text = "I understand you are waiting for your refund. According to our policy, it will take 3-5 business days to process to your original payment method. Is there anything else I can assist you with?"
        else:
            ai_text = "Thank you for reaching out! Based on our documentation, I would be happy to help you with your account. Could you please provide your order number?"
            
        return ChatResponse(response=ai_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))