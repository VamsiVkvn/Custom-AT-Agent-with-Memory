from fastapi import FastAPI
from pydantic import BaseModel
from assistant import generate_assistant_response, update_long_term_memory

app = FastAPI()

class Message(BaseModel):
    prompt: str
    session_id: str

@app.post("/chat")
async def chat_endpoint(message: Message):
    """API endpoint to get assistant response and update memory."""
    response_text = generate_assistant_response(message.prompt, [])
    update_long_term_memory(message.prompt, response_text, message.session_id)
    return {"response": response_text}
