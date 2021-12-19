from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

class TrainingRecord(BaseModel):
    passenger_id: int
    survived: bool
    pclass: int
    name: str
    sex: str
    age: int
    sibsp: int
    parch: int
    ticket: str
    fare: float
    cabin: str
    embarked: str

class TestRecord(BaseModel):
    passenger_id: int
    pclass: int
    name: str
    sex: str
    age: int
    sibsp: int
    parch: int
    ticket: str
    fare: float
    cabin: str
    embarked: str

@app.get("/")
async def read_root():
    return {"Hello": "World"}

# select
@app.get("/trainingrecord/{passenger_id}")
async def read_item( passenger_id: int ):
    cmd = "grep -e '^" + str(passenger_id) + ",' ../data/train.csv"
    stream = os.popen( cmd )
    out = stream.read()
    if len(out)==0:
        raise HTTPException(status_code=404, detail="Not found")
    else:
        a = out.split(',')
        return { "out": out, "passenger_id": a[0], "pclass": a[1], "name": a[2], "sex": a[3], "age": a[4], "sibsp": a[5], "parch": a[6], "ticket": a[7], "fare": a[8], "cabin": a[9], "embarked": a[10] }

# delete
#@app.delete("/trainingrecord/{passenger_id}")
#async def read_item( passenger_id: int, survived: bool, pclass: int, name: str, sex: str, age: int, sibsp: int, parch: int, ticket: str, fare: float, cabin: str, embarked: str ):
#    return {}
# insert
#@app.post("/trainingrecord/{passenger_id}")
#async def read_item( passenger_id: int, survived: bool, pclass: int, name: str, sex: str, age: int, sibsp: int, parch: int, ticket: str, fare: float, cabin: str, embarked: str ):
#    return {"item_id": item_id, "q": q}
# update
#@app.put("/trainingrecord/{passenger_id}")
#async def read_item( passenger_id: int, survived: bool, pclass: int, name: str, sex: str, age: int, sibsp: int, parch: int, ticket: str, fare: float, cabin: str, embarked: str ):
#    return {"item_id": item_id, "q": q}


