from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"Message:" "Model metrics are being collected and available at /metrics"}

if __name__ == "_main__":
    uvicorn.run("metrics_collector:app", host = "0.0.0.0", port = 8000, reload = True)