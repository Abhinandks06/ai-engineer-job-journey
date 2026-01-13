from fastapi import FastAPI

app = FastAPI(title="AI Engineer Job Journey")

@app.get("/")
def root():
    return {"message": "FastAPI is running successfully"}
