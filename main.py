from typing import Union
from fastapi import FastAPI
from vertex_ai import prompt_ai

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/Jidrid_ai")
def prom_jidrid(promp = ""):
    result = prompt_ai(promp)
    return result
