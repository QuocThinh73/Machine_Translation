from fastapi import FastAPI
from translation import translate, models
from pydantic import BaseModel

app = FastAPI()


class TranslationRequest(BaseModel):
    text: str
    model: str


class TranslationResponse(BaseModel):
    translated_text: str


@app.post("/translation", response_model=TranslationResponse)
def get_translation(request: TranslationRequest):
    translated_text = translate(request.text, request.model)
    return TranslationResponse(translated_text=translated_text)


@app.get("/models")
def get_models():
    return {"models": models}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
