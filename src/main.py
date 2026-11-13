"""Momir Basic Printer - Headless hardware-driven appliance for printing Magic: The Gathering cards.

This module provides an event-driven control loop for:
- Printing a random card at the selected CMC via thermal printer on short press
- Cancelling an active operation via long press
- Refreshing card data on startup when needed
"""

import configparser
import enum
import logging
import signal
import string
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from flask import Flask, render_template, request

#Create Flask app
flaskApp = Flask(__name__)

# Configuration
config = configparser.ConfigParser()
config_file = Path(__file__).resolve().with_name('config.ini')

if not config_file.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_file}")

config.read(config_file)

# Validate required sections exist
required_sections = ['APP', 'FILESYSTEM', 'HARDWARE', 'LOGGING', 'PRINTER', 'SCRYFALL']
missing_sections = [
    section for section in required_sections if section not in config]
if missing_sections:
    raise ValueError(
        f"Missing required configuration sections: {', '.join(missing_sections)}")

app_config = config['APP']
filesystem_config = config['FILESYSTEM']
hardware_config = config['HARDWARE']
logging_config = config['LOGGING']
printer_config = config['PRINTER']
scryfall_config = config['SCRYFALL']

# Logging setup
logger = logging.getLogger('momir.main')
logging.basicConfig(
    level=logging_config.get('log_level').upper(),
    format=logging_config.get('log_format'),
    datefmt=logging_config.get('log_date_format'),
    stream=sys.stdout,
    force=True,
)


class AppState(enum.Enum):
    IDLE = "idle"
    FETCHING = "fetching"
    PRINTING = "printing"


class MomirApp:
    """Main application controller for the headless Momir Basic Printer."""

    def __init__(self) -> None:
        self._state = AppState.IDLE
        self._state_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # CMC bounds from config
        self._cmc_min: int = hardware_config.getint('cmc_min', fallback=0)
        self._cmc_max: int = hardware_config.getint('cmc_max', fallback=16)
        self._cmc: int = self._cmc_min

        # Services (set up in initialize)
        self.printer: Optional[Any] = None
        self.scryfall: Optional[Any] = None

        # App UI/runtime text and limits
        self._booting_status: str = app_config.get('booting_status', fallback='Booting...')
        self._ready_status: str = app_config.get('ready_status', fallback='Ready')
        self._refreshing_status: str = app_config.get('refreshing_status', fallback='Refreshing...')
        self._fetching_status: str = app_config.get('fetching_status', fallback='Fetching...')
        self._printing_status: str = app_config.get('printing_status', fallback='Printing...')
        self._cancelled_status: str = app_config.get('cancelled_status', fallback='Cancelled')
        self._error_status: str = app_config.get('error_status', fallback='Error')
        self._reset_status: str = app_config.get('reset_status', fallback='Reset')
        self._services_unavailable_status: str = app_config.get(
            'services_unavailable_status', fallback='Services N/A')
        self._done_status: str = app_config.get('done_status', fallback='Done!')
        self._done_status_seconds: float = app_config.getfloat(
            'done_status_seconds', fallback=2.0)
        self._no_cmc_status_template: str = app_config.get(
            'no_cmc_status_template', fallback='No CMC {cmc}')
        self._no_name_status_template: str = app_config.get(
            'no_name_status_template', fallback='No card named {name}')
        self._shutdown_join_timeout_seconds: float = app_config.getfloat(
            'shutdown_join_timeout_seconds', fallback=5)

    def _clamp_cmc(self, cmc: int) -> int:
        return max(self._cmc_min, min(cmc, self._cmc_max))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize printer and scryfall services."""

        # Printer
        try:
            from printer import Printer
            self.printer = Printer(printer_config, filesystem_config, hardware_config)
            logger.info("Printer service initialized.")
        except Exception as exc:
            self.printer = None
            logger.error(f"Printer service unavailable: {exc}")

        # Scryfall
        try:
            from scryfall import Scryfall
            self.scryfall = Scryfall(
                scryfall_config, filesystem_config, printer_config)
            logger.info("Scryfall service initialized.")
        except Exception as exc:
            self.scryfall = None
            logger.error(f"Scryfall service unavailable: {exc}")

        # Auto-refresh if needed
        if self.scryfall is not None and self.scryfall.needs_refresh():
            self._set_status(self._refreshing_status)
            try:
                self.scryfall.refresh_card_data(force_full_refresh=False)
                logger.info("Card data refresh complete.")
            except Exception as exc:
                logger.error(f"Card data refresh failed: {exc}")

        self._set_status(self._ready_status)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _set_state(self, state: AppState) -> None:
        with self._state_lock:
            self._state = state

    def _get_state(self) -> AppState:
        with self._state_lock:
            return self._state

    def _set_status(self, text: str) -> None:
        logger.debug(f"Status: {text}")

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_held(self) -> None:
        """Fires when button is held past hold_time — treat as long press."""
        self._held_fired = True
        self._on_long_press()

    def _on_released(self) -> None:
        """Fires on button release. If hold wasn't triggered, treat as short press."""
        if self._held_fired:
            self._held_fired = False
            return
        self._on_short_press()

    def _on_short_press(self) -> None:
        """Short press: fetch and print a random card at current CMC."""
        if self._get_state() != AppState.IDLE:
            logger.debug("Ignoring short press — operation in progress.")
            return

        if self.scryfall is None or self.printer is None:
            self._set_status(self._services_unavailable_status)
            return

        self._cancel_event.clear()
        self._worker_thread = threading.Thread(
            target=self._fetch_and_print, args=(self._cmc,), daemon=True)
        self._worker_thread.start()

    def _on_long_press(self) -> None:
        """Long press: cancel active operation or reset to idle."""
        if self._get_state() != AppState.IDLE:
            logger.info("Long press — cancelling active operation.")
            self._cancel_event.set()
        else:
            self._cmc = self._cmc_min

    # ------------------------------------------------------------------
    # Worker (runs in background thread)
    # ------------------------------------------------------------------

    def _fetch_and_print(self, cmc: int) -> None:
        try:
            # --- fetch ---
            self._set_state(AppState.FETCHING)
            self._set_status(self._fetching_status)
            if self._cancel_event.is_set():
                self._set_status(self._cancelled_status)
                return

            card = self.scryfall.get_random_card_by_cmc(cmc)

            if card is None:
                self._set_status(self._no_cmc_status_template.format(cmc=cmc))
                return

            if self._cancel_event.is_set():
                self._set_status(self._cancelled_status)
                return

            # --- print ---
            self._set_state(AppState.PRINTING)
            card_name = card.get("name", "Unknown")
            self._set_status(self._printing_status)
            logger.info(f"Printing: {card_name}")

            self.printer.print_card(card, cancel_event=self._cancel_event)
            if self._cancel_event.is_set():
                self._set_status(self._cancelled_status)
                logger.info(f"Printing cancelled: {card_name}")
                return

            self._set_status(self._done_status)
            if self._done_status_seconds > 0:
                time.sleep(self._done_status_seconds)
            self._set_status(self._ready_status)
            logger.info(f"Printed: {card_name}")

        except Exception as exc:
            if self._cancel_event.is_set():
                logger.info(f"Fetch/print cancelled: {exc}")
                self._set_status(self._cancelled_status)
            else:
                logger.error(f"Fetch/print error: {exc}")
                self._set_status(self._error_status)
        finally:
            self._set_state(AppState.IDLE)

    def _fetch_and_print(self, name: string) -> None:
        try:
            # --- fetch ---
            self._set_state(AppState.FETCHING)
            self._set_status(self._fetching_status)
            if self._cancel_event.is_set():
                self._set_status(self._cancelled_status)
                return

            card = self.scryfall.get_card_by_name(name)

            if card is None:
                self._set_status(self._no_name_status_template.format(name=name))
                return

            if self._cancel_event.is_set():
                self._set_status(self._cancelled_status)
                return

            # --- print ---
            self._set_state(AppState.PRINTING)
            card_name = card.get("name", "Unknown")
            self._set_status(self._printing_status)
            logger.info(f"Printing: {card_name}")

            self.printer.print_card(card, cancel_event=self._cancel_event)
            if self._cancel_event.is_set():
                self._set_status(self._cancelled_status)
                logger.info(f"Printing cancelled: {card_name}")
                return

            self._set_status(self._done_status)
            if self._done_status_seconds > 0:
                time.sleep(self._done_status_seconds)
            self._set_status(self._ready_status)
            logger.info(f"Printed: {card_name}")

        except Exception as exc:
            if self._cancel_event.is_set():
                logger.info(f"Fetch/print cancelled: {exc}")
                self._set_status(self._cancelled_status)
            else:
                logger.error(f"Fetch/print error: {exc}")
                self._set_status(self._error_status)
        finally:
            self._set_state(AppState.IDLE)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------


    def shutdown(self) -> None:
        """Clean up hardware resources and exit."""
        logger.info("Shutting down...")
        self._cancel_event.set()

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=self._shutdown_join_timeout_seconds)

        if self.printer is not None:
            self.printer.cleanup()

        self._shutdown_event.set()
        logger.info("Momir Basic Printer stopped.")


def main() -> None:
    flaskApp.run(host="192.168.1.65", port=5000, debug=True)   



app = MomirApp()
app.initialize()

# Route to display the HTML form
@flaskApp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Route to process the submitted form
@flaskApp.route("/read_form", methods=["POST"])
def read_form():

    # Retrieve form data
    name = request.form.get("cardName")

    app._fetch_and_print(name)

    # Return the submitted data
    return {
        "name": name
    }


if __name__ == "__main__":
    main()


