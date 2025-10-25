from fastapi import FastAPI, Request
from pydantic import BaseModel
from algo import PokerAI, OpponentModel

app = FastAPI()
om = OpponentModel()
ai = PokerAI(om)

class GameState(BaseModel):
    hero: str
    hole_cards: list[str]
    community_cards: list[str]
    pot: float
    to_call: float
    stage: str
    players: list[str]
    stacks: dict[str, float] = {}
    position: str = "mp"
    min_raise: float = 0

@app.post("/decide")
async def decide(game: GameState):
    result = ai.decide(game.dict())
    return result

# Optional: for local testing
@app.get("/")
def home():
    return {"status": "AI Ready"}
