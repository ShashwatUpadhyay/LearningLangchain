from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, EmailStr, Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    # max_tokens=None, 
    # timeout=None,
    # max_retries=2
)

#schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes siscussed in the review")
    summary: str = Field(description="A breaf summarry of the review")
    sentiment: Literal['neg','pos','neu'] = Field(description="return the sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None,description="Write down all the pros mentioned in the review")
    cons: Optional[list[str]] = Field(default=None,description="Write down all the cons mentioned in the review")
    name: Optional[str] = Field(default=None, description="Write down the name of the person who wrote the review only if given") 

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
                                 I’ve been using these headphones for about a month now, and overall, I’m quite impressed. The sound quality is phenomenal — highs are crisp, mids are clear, and the bass really gives depth to my favorite tracks. Listening to different genres, from classical to EDM, the experience feels immersive, and I often forget I’m wearing them because the audio is so well-balanced. The active noise cancellation is effective as well, letting me focus in noisy environments, whether at home or in a busy café."

"Comfort is another strong point. The ear cups are soft and the headband adjusts nicely, so even during long listening sessions, I don’t feel any strain. I particularly appreciate the lightweight design, which makes it easy to wear while commuting or even during work. The build feels premium — solid enough to withstand daily use without feeling flimsy — yet stylish, so it doesn’t look out of place if I step outside."

"Battery life is decent, but not exceptional. On heavier usage days, I notice I have to recharge more often than I’d like. It’s not a dealbreaker, but it’s something to keep in mind if you’re planning long trips without access to a charger. The controls on the headphones are intuitive, though sometimes I accidentally skip tracks when adjusting volume. Pairing with multiple devices is easy, which is convenient when switching between my laptop and phone."

"Overall, these headphones offer a fantastic listening experience with excellent sound, comfort, and design. Minor inconveniences like the battery life and sensitive controls are worth overlooking given the quality of audio and ease of use. I’d definitely recommend them to anyone looking for a well-rounded set of headphones that feel premium without completely breaking the bank.
                                 """)
print(result.dict())