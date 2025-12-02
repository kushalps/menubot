Menu Chatbot
uses locally hosted phi3 model

Steps:
1. > Install dependencies:
   > pip install -r requirements.txt

2. > Download ollama:
   > curl -fsSL https://ollama.com/install.sh | sh

3. > Downloaf phi3 model:
   > ollama pull phi3

4. > Start service in a different shell:
   > ollama serve

5. > Run chatbot:
   > python3 chatbot.py
