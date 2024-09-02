import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.document_processor import query_to_answer

from fastapi import FastAPI, APIRouter
from pydantic_models.models import UserPromptRequest, ModelResponse
# import the model
# query the model
# return the response from the model

app = FastAPI()
router = APIRouter()

memory = []

@router.post("/ai/prompt", response_model=ModelResponse)
def create_item(user_prompt_request: UserPromptRequest):
    # ai_response = model.query(user_pormpt_request.prompt)
    print(user_prompt_request.prompt)
    ai_response = query_to_answer(query=user_prompt_request.prompt, memory=memory)
    response = ModelResponse(
        user_prompt=user_prompt_request,
        ai_response=ai_response
    )
    return response

app.include_router(router)