from pydantic import BaseModel, Field

class ReviewRequest(BaseModel):
	txt: str = Field(..., min_lenght=3, description="Review text to classify")


class ReviewResponse(BaseModel):
	label: str
	fake_probability: float
	suspicious_signals: list[str]