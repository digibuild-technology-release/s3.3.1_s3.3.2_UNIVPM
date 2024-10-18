import asyncio
from aiogram import Bot, types
from datetime import datetime, time, timezone, timedelta
from flask import Flask
import threading

# Token del tuo bot Telegram
TOKEN = '7384804986:AAHcsA7ToR5QbUF4U-BkYK3WOc9luv5I3Ns'

# ID del chat dove inviare il reminder (può essere il tuo chat ID personale o di un gruppo)
CHAT_ID = '-1002230148964'

app = Flask('')

@app.route('/')
def home():
    return "Bot is running!"

def run():
    app.run(host='0.0.0.0', port=8080)

# Funzione per verificare se è il momento di inviare il reminder
def is_reminder_time():
    now = datetime.now(timezone(timedelta(hours=2)))
    current_time = now.time()
    return current_time >= time(7, 0) and current_time <= time(22, 0)

# Funzione per inviare una notifica push
async def send_push_notification():
    bot = Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text="Ricordati di votare!\nClicca qui: https://digibuild-demo.eu/digibuild/", disable_notification=False)

# Funzione principale asincrona per eseguire il bot
async def main():
    while True:
        try:
            # Controlla se è il momento consentito per inviare il reminder
            if is_reminder_time():
                # Invia la notifica push
                await send_push_notification()

            # Aspetta 1 ora prima di inviare la prossima notifica
            await asyncio.sleep(3600)  # 3600 secondi = 1 ora
        except Exception as e:
            print(f'Errore durante l\'invio della notifica: {str(e)}')
            await asyncio.sleep(60)  # Pausa di 1 minuto prima di riprovare

# Esegui il bot
if __name__ == "__main__":
    t = threading.Thread(target=run)
    t.start()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
