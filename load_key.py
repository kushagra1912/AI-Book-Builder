from dotenv import load_dotenv
import os

load_dotenv()


def load_key():
    return os.getenv("GOOGLE_API_KEY")
