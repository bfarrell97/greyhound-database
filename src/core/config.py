"""Configuration file for API keys and application settings.

This module contains all API credentials and configuration constants
required for the Greyhound Racing Analysis System.

⚠️ Security Warning:
    API keys and passwords should be stored in environment variables
    or a separate .env file (not version controlled). This file contains
    hardcoded credentials for convenience during development only.

Attributes:
    TOPAZ_API_KEY (str): API key for Topaz form data service
    BETFAIR_APP_KEY (str): Betfair application key for API access
    BETFAIR_USERNAME (str): Betfair account username
    BETFAIR_PASSWORD (str): Betfair account password
    BETFAIR_CERT_PATH (str): Path to SSL certificate for non-interactive login
    BETFAIR_KEY_PATH (str): Path to SSL private key for non-interactive login
    DISCORD_WEBHOOK_URL (str): Discord webhook URL for notifications

Example:
    >>> from src.core.config import BETFAIR_APP_KEY, DISCORD_WEBHOOK_URL
    >>> print(f"Using app key: {BETFAIR_APP_KEY[:8]}...")
    Using app key: Gb4rI9sB...
"""

import os
from typing import Final

# Topaz API Key
TOPAZ_API_KEY: Final[str] = "313c5027-4e3b-4f5b-a1b4-3608153dbaa3"

# Betfair API Credentials
BETFAIR_APP_KEY: Final[str] = "Gb4rI9sBY2mUauRD"
BETFAIR_USERNAME: Final[str] = "bfarrell97@hotmail.com"
BETFAIR_PASSWORD: Final[str] = "Bradams900!"

# SSL Certificates for Non-Interactive Login
BASE_DIR: Final[str] = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
BETFAIR_CERT_PATH: Final[str] = os.path.join(BASE_DIR, 'certs', 'client-2048.crt')
BETFAIR_KEY_PATH: Final[str] = os.path.join(BASE_DIR, 'certs', 'client-2048.key')

# Discord Notification Webhook
DISCORD_WEBHOOK_URL: Final[str] = (
    "https://discordapp.com/api/webhooks/1451873744506195970/"
    "e1barMNV_0piEmCke-A0vzu6qRBmMfjsrXz0K8YIVlNVeIq027yj1ipHu_2vcQHFJ3WN"
)
