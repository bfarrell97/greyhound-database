"""
Configuration file for API keys and settings
"""

# Topaz API Key
TOPAZ_API_KEY = "313c5027-4e3b-4f5b-a1b4-3608153dbaa3"

# Betfair API Credentials
BETFAIR_APP_KEY = "Gb4rI9sBY2mUauRD"
BETFAIR_USERNAME = "bfarrell97@hotmail.com"
BETFAIR_PASSWORD = "Bradams900!"

import os

# SSL Certificates for Non-Interactive Login
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BETFAIR_CERT_PATH = os.path.join(BASE_DIR, 'certs', 'client-2048.crt')
BETFAIR_KEY_PATH = os.path.join(BASE_DIR, 'certs', 'client-2048.key')

# Discord Notification Webhook (User to populate)
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1451873744506195970/e1barMNV_0piEmCke-A0vzu6qRBmMfjsrXz0K8YIVlNVeIq027yj1ipHu_2vcQHFJ3WN"
