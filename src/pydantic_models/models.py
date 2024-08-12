from pydantic import BaseModel

class UserPromptRequest(BaseModel):
    prompt: str


class ModelResponse(BaseModel):
    user_prompt: UserPromptRequest
    ai_response: str