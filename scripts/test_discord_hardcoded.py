
import sys
import os
sys.path.append(os.getcwd())
from src.core.config import DISCORD_WEBHOOK_URL
from src.utils.discord_notifier import DiscordNotifier
import time

print(f"Loaded Webhook URL: '{DISCORD_WEBHOOK_URL}'")

if not DISCORD_WEBHOOK_URL:
    print("ERROR: Webhook URL is empty!")
else:
    print("Sending test notification...")
    DiscordNotifier.send_notification(
        title="🕵️ Debug Test", 
        message="If you see this, the Webhook URL is correct and the Notifier class works.",
        color=0x00ffff
    )
    print("Notification sent (threaded). Waiting 3 seconds...")
    time.sleep(3)
    print("Done.")
