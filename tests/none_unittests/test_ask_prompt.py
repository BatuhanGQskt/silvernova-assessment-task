# This file is used for manual testing of the ask_prompt function. One can also use manual testing on test_integration.py
# Basically, my playground to check anything manually. No expectation of unittests here.

import sys
from pathlib import Path

# Add parent directory to path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))
from src.api import execute_prompt

from src.operations.ask import LLMAsker

def ask_prompt(message: str):
    response = execute_prompt(message)
    
    return response["response"]

def test_prompt_Besamgt(message: str):
    correct_answer = "1,213.92 EUR VAT"

    asker = LLMAsker(max_tokens=100000)

    print("ASKING THE QUESTION...")
    provided_answer = asker.ask(message)
    print("Provided answer:", provided_answer)
    print("ASKING...")

    prompt = "Können Sie die folgenden beiden Antworten vergleichen und mir sagen, ob sie gleich oder unterschiedlich sind?\n\n"
    prompt += f"Richtige Antwort: {correct_answer}\n"
    prompt += f"Antwort bereitgestellt: {provided_answer}\n"
    prompt += "Bitte geben Sie nur dann 'true' zurück, wenn die angegebene Antwort korrekt ist, und nur dann 'false', wenn die Antworten unterschiedlich sind."

    response = ask_prompt(prompt)

    print(response)


# ask_prompt("What is the capital of France?")

test_prompt_Besamgt("Was ist die Betrag gesamt EUR für die Haufe Gruppe?")