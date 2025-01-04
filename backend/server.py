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
async def get_translation(request: TranslationRequest):
    translated_text = translate(request.text, request.model)
    return TranslationResponse(translated_text=translated_text)


@app.get("/models")
async def get_models():
    return {"models": list(models.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
