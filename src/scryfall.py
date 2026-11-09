"""Scryfall API client for downloading and managing Magic: The Gathering card data."""

import json
import logging
import random
import shutil
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Set

import gzip
import requests
from PIL import Image

logger = logging.getLogger('momir.scryfall')


class Scryfall:
    """Scryfall API client for downloading and managing Magic: The Gathering card data.

    This class provides methods for:
    - Downloading card data from Scryfall's bulk API
    - Processing and storing card information locally
    - Managing incremental updates to minimize bandwidth usage
    - Filtering cards based on Momir Basic format rules
    """

    # File and format constants
    METADATA_FILENAME = "metadata.json"
    CARD_FILE_EXTENSION = ".json"
    ART_FILE_EXTENSION = ".jpg"

    # Processing constants
    PROGRESS_LOG_INTERVAL = 1000
    MONOCHROME_MODE = "1"
    HTTP_OK = 200
    REQUEST_TIMEOUT = 30
    BULK_DOWNLOAD_URI_FIELD = "jsonl_download_uri"  # Scryfall bulk data download field
    IMAGE_ACCEPT_HEADER = "image/*"  # Accept header for image CDN requests

    # Momir Basic validation constants
    PAPER_FORMAT = "paper"
    CREATURE_TYPE = "creature"

    def __init__(self, scryfall_config, filesystem_config, printer_config=None) -> None:
        """Initialize Scryfall client with configuration.

        Args:
            scryfall_config: Configuration section for Scryfall API settings
            filesystem_config: Configuration section for filesystem paths
            printer_config: Optional configuration section for printer settings,
                used to determine whether card art is needed at all
        """
        # API configuration
        self.base_url: str = scryfall_config.get('base_url')
        self.bulk_data_endpoint: str = scryfall_config.get(
            'bulk_data_endpoint')
        self.header_accept: str = scryfall_config.get('header_accept')
        self.header_user_agent: str = scryfall_config.get('header_user_agent')
        self.header_accept_encoding: str = scryfall_config.get(
            'header_accept_encoding')

        # Processing configuration
        self.request_delay_seconds: float = scryfall_config.getfloat(
            'request_delay_seconds')
        self.max_retries: int = scryfall_config.getint('max_retries')
        self.art_width_px: int = scryfall_config.getint('art_width_px')

        # Card art is only downloaded and stored when the printer will use it
        self.card_art_enabled: bool = (
            printer_config.getboolean('card_art_enabled', fallback=True)
            if printer_config is not None else True
        )

        # Filter configuration
        self.excluded_sets: List[str] = [
            s.strip() for s in scryfall_config.get('excluded_sets').replace('\n', '').split(',') if s.strip()
        ]
        self.excluded_layouts: List[str] = [
            layout.strip() for layout in scryfall_config.get('excluded_layouts').replace('\n', '').split(',') if layout.strip()
        ]

        # Filesystem paths (use __file__ for reliability)
        base_path = Path(__file__).resolve().parent.parent
        self.cards_path: Path = base_path / filesystem_config.get('cards_path')
        self.art_path: Path = base_path / filesystem_config.get('art_path')
        self.default_card_art_path: Path = base_path / \
            filesystem_config.get('default_card_art_path')
        self.access_rights: int = int(
            filesystem_config.get('access_rights'), 0)

        # Ensure runtime directories exist so app can boot on a fresh install.
        self.cards_path.mkdir(parents=True, exist_ok=True, mode=self.access_rights)
        if self.card_art_enabled:
            self.art_path.mkdir(parents=True, exist_ok=True, mode=self.access_rights)

        # Validate critical configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate critical configuration values on initialization.

        Raises:
            ValueError: If critical configuration is missing or invalid
        """
        if not self.base_url or not self.bulk_data_endpoint:
            raise ValueError("Scryfall API URL configuration is missing")

        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}")

        if self.art_width_px <= 0:
            raise ValueError(
                f"art_width_px must be positive, got {self.art_width_px}")

        if self.card_art_enabled and not self.default_card_art_path.exists():
            logger.warning(
                f"Default card art not found at {self.default_card_art_path}. "
                "Fallback art will be generated when needed."
            )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Scryfall(base_url='{self.base_url}', "
                f"cards_path='{self.cards_path}', "
                f"max_retries={self.max_retries})")

    # ===== Public API Methods =====

    def get_valid_cmcs(self) -> List[int]:
        """Get a sorted list of all CMC directories that exist.

        Returns:
            Sorted list of CMC values (integers)
        """
        if not self.cards_path.exists():
            return []

        cmc_dirs = [int(d.name) for d in self.cards_path.iterdir()
                    if d.is_dir() and d.name.isdigit()]
        return sorted(cmc_dirs)

    def get_total_card_count(self) -> int:
        """Count the total number of card JSON files in CMC directories.

        Returns:
            Total number of card files
        """
        if not self.cards_path.exists():
            return 0

        total_count = 0
        for cmc_dir in self.cards_path.iterdir():
            if not (cmc_dir.is_dir() and cmc_dir.name.isdigit()):
                continue

            total_count += sum(
                1
                for card_file in cmc_dir.iterdir()
                if card_file.is_file() and card_file.suffix == self.CARD_FILE_EXTENSION
            )

        return total_count

    def get_card_count_by_cmc(self, cmc: int) -> int:
        """Count the number of cards with a specific CMC.

        Args:
            cmc: Converted mana cost to count

        Returns:
            Number of cards with the specified CMC
        """
        path = self.cards_path / str(cmc)
        if not path.exists():
            return 0
        return sum(
            1
            for card in path.iterdir()
            if card.is_file() and card.suffix == self.CARD_FILE_EXTENSION
        )

    def get_card_art_by_card_id(self, card_id: str) -> Optional[Image.Image]:
        """Load card art from disk by card ID.

        Args:
            card_id: Unique Scryfall card ID

        Returns:
            PIL Image object or None if not found

        Raises:
            OSError: If image file exists but cannot be opened
        """
        card_art_path = self._get_art_path(card_id)

        if not card_art_path.is_file():
            logger.warning(f"Card art not found: {card_id}")
            return None

        try:
            return Image.open(card_art_path)
        except OSError as e:
            logger.error(f"Failed to open card art {card_id}: {e}")
            raise

    def get_random_card_by_cmc(self, cmc: int) -> Optional[Dict[str, Any]]:
        """Get a random card with the specified CMC.

        Args:
            cmc: Converted mana cost to search for (must be non-negative)

        Returns:
            Card data dictionary or None if no cards found

        Raises:
            ValueError: If cmc is negative
            json.JSONDecodeError: If card JSON is malformed
        """
        if cmc < 0:
            raise ValueError(f"CMC must be non-negative, got {cmc}")

        path = self.cards_path / str(cmc)

        if not path.exists():
            logger.warning(f"CMC directory not found: {cmc}")
            return None

        cards = [card for card in path.iterdir() if card.is_file()]

        if not cards:
            logger.warning(f"No cards found for CMC: {cmc}")
            return None

        random_card_path = random.choice(cards)
        with open(random_card_path, 'r', encoding='utf-8') as card_file:
            return json.load(card_file)

    def get_card_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Search for a card by name across all CMC directories.

        Args:
            name: Name of the card to search for (case-insensitive)
        """
        name_lower = name.lower()

        for cmc_dir in self.cards_path.iterdir():
            if not (cmc_dir.is_dir() and cmc_dir.name.isdigit()):
                continue

            for card_file in cmc_dir.iterdir():
                if card_file.is_file() and card_file.suffix == self.CARD_FILE_EXTENSION:
                    with open(card_file, 'r', encoding='utf-8') as f:
                        card_data = json.load(f)
                        if card_data.get('name', '').lower() == name_lower:
                            return card_data

        logger.warning(f"Card not found by name: {name}")
        return None
            

    def is_valid_momir_basic_card(self, card: Dict[str, Any]) -> bool:
        """Check if a card is valid for Momir Basic format.

        Validates based on:
        - Card layout (excludes special layouts like tokens, vanguards)
        - Set type (excludes memorabilia, promo sets, etc.)
        - Paper availability (must be available in paper format)
        - Card type (must be a creature)

        Args:
            card: Card data dictionary from Scryfall

        Returns:
            True if card is valid for Momir Basic, False otherwise
        """
        # Check layout
        if card.get('layout') in self.excluded_layouts:
            return False

        # Check set type
        if card.get('set_type') in self.excluded_sets:
            return False

        # Check if available in paper format
        if self.PAPER_FORMAT not in card.get('games', []):
            return False

        # Check if it's a creature (handle double-faced cards)
        if 'card_faces' in card:
            front_type = card['card_faces'][0].get('type_line', '').lower()
            return self.CREATURE_TYPE in front_type

        type_line = card.get('type_line', '').lower()
        return self.CREATURE_TYPE in type_line

    def download_bulk_metadata(self) -> Dict[str, Any]:
        """Download bulk data metadata from Scryfall.

        Returns:
            Dictionary containing bulk data metadata

        Raises:
            requests.RequestException: If the request fails
        """
        headers = self._get_request_headers()
        url = f"{self.base_url}{self.bulk_data_endpoint}"

        try:
            response = requests.get(
                url, headers=headers, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching bulk metadata from {url}: {e}")
            raise

    def filter_bulk_data_by_cmc(self, bulk_data: List[Dict[str, Any]], cmc: float) -> List[Dict[str, Any]]:
        """Filter a list of cards by CMC.

        Args:
            bulk_data: List of card dictionaries
            cmc: Converted mana cost to filter by

        Returns:
            Filtered list of cards matching the CMC
        """
        return [card for card in bulk_data if card.get('cmc') == cmc]

    def delete_directory(self, path: Path) -> None:
        """Delete a directory and all its contents.

        Args:
            path: Path to directory to delete
        """
        path_obj = Path(path)
        if path_obj.exists():
            shutil.rmtree(path_obj, ignore_errors=True)
            logger.debug(f"Deleted directory: {path}")

    def create_directory(self, path: Path) -> None:
        """Create a directory if it doesn't already exist.

        Args:
            path: Path to directory to create
        """
        Path(path).mkdir(parents=True, exist_ok=True, mode=self.access_rights)

    def save_card(self, path: Path, card: Dict[str, Any]) -> None:
        """Save card data as JSON to the filesystem.

        Args:
            path: Filesystem path to save to
            card: Card data dictionary
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(card, f, indent=2, ensure_ascii=False)

    # ===== Private Helper Methods =====

    # Configuration and Utility Helpers

    def _get_request_headers(self, include_encoding: bool = False) -> Dict[str, str]:
        """Build request headers for Scryfall API calls.

        Args:
            include_encoding: Whether to include Accept-Encoding header

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            'Accept': self.header_accept,
            'User-Agent': self.header_user_agent
        }
        if include_encoding:
            headers['Accept-Encoding'] = self.header_accept_encoding
        return headers

    def _get_image_request_headers(self) -> Dict[str, str]:
        """Build request headers for card art downloads from Scryfall's image CDN.

        The image CDN rejects any request carrying a default HTTP library
        User-Agent with HTTP 400 ("generic_user_agent"), so art downloads must
        identify the application just like API calls do.

        Returns:
            Dictionary of HTTP headers
        """
        return {
            'Accept': self.IMAGE_ACCEPT_HEADER,
            'User-Agent': self.header_user_agent
        }

    # Path Resolution Helpers

    def _get_card_path(self, card_id: str, cmc: int) -> Path:
        """Get the filesystem path for a card JSON file.

        Args:
            card_id: Unique Scryfall card ID
            cmc: Converted mana cost

        Returns:
            Path object for card JSON file
        """
        return self.cards_path / str(cmc) / f"{card_id}{self.CARD_FILE_EXTENSION}"

    def _get_art_path(self, card_id: str) -> Path:
        """Get the filesystem path for card art.

        Args:
            card_id: Unique Scryfall card ID

        Returns:
            Path object for card art file
        """
        return self.art_path / f"{card_id}{self.ART_FILE_EXTENSION}"

    # Card Data Extraction Helpers

    def _get_card_art_uri(self, card: Dict[str, Any]) -> Optional[str]:
        """Extract art crop URI from card data.

        Args:
            card: Card data dictionary

        Returns:
            Art crop URI string or None if not found
        """
        return (card.get('card_faces', [{}])[0].get('image_uris', {}).get('art_crop')
                or card.get('image_uris', {}).get('art_crop'))

    # Image Processing Helpers

    def _process_image(self, image_bytes: bytes) -> Image.Image:
        """Process image: resize and convert to monochrome.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Processed PIL Image object
        """
        img = Image.open(BytesIO(image_bytes))

        # Calculate new dimensions maintaining aspect ratio
        w_percent = self.art_width_px / float(img.size[0])
        h_size = int(float(img.size[1]) * w_percent)

        # Resize and convert to monochrome
        img = img.resize((self.art_width_px, h_size), Image.Resampling.LANCZOS)
        return img.convert(self.MONOCHROME_MODE)

    # Data Processing and Cleanup Helpers

    def _ensure_fallback_art(self, path: Path) -> None:
        """Write fallback art to disk when remote art download fails.

        Prefers configured default art image, but if it is missing, generates a
        blank monochrome placeholder so processing can continue.

        Args:
            path: Filesystem path for fallback art output
        """
        if self.default_card_art_path.exists():
            shutil.copy(self.default_card_art_path, path)
            return

        placeholder = Image.new(self.MONOCHROME_MODE, (self.art_width_px, self.art_width_px), 1)
        placeholder.save(path)
        logger.warning(f"Generated placeholder fallback art: {path.name}")

    def _log_progress(self, stats: Dict[str, int], is_full_refresh: bool) -> None:
        """Log progress during card processing.

        Args:
            stats: Dictionary containing processing statistics
            is_full_refresh: Whether this is a full refresh operation
        """
        if is_full_refresh:
            logger.info(
                f"Processed {stats['total_processed']} cards. "
                f"Found {stats['total_creatures']} valid creatures so far."
            )
        else:
            logger.info(
                f"Processed {stats['total_processed']} cards "
                f"({stats['new_cards']} new, {stats['skipped_cards']} existing)."
            )

    def save_card_art(self, path: Path, card_art_uri: str) -> None:
        """Download and save card art with automatic retry logic.

        Downloads card art from the given URI, resizes and converts to monochrome,
        then saves to disk. If download fails after all retries, uses default art.

        Args:
            path: Filesystem path to save art to
            card_art_uri: URI of the card art to download
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(
                    card_art_uri, headers=self._get_image_request_headers(),
                    timeout=self.REQUEST_TIMEOUT)

                if response.status_code == self.HTTP_OK:
                    img = self._process_image(response.content)
                    img.save(path)
                    logger.debug(f"Saved card art: {path.name}")
                    sleep(self.request_delay_seconds)
                    return

                # Non-200 status code
                if attempt < self.max_retries:
                    logger.warning(
                        f"Download failed (HTTP {response.status_code}). "
                        f"Retry {attempt + 1}/{self.max_retries}: {card_art_uri}"
                    )
                    sleep(self.request_delay_seconds)
                else:
                    logger.warning(
                        f"All retries exhausted. Using default art: {card_art_uri}")
                    self._ensure_fallback_art(path)
                    return

            except requests.RequestException as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Request exception ({type(e).__name__}). "
                        f"Retry {attempt + 1}/{self.max_retries}: {card_art_uri}"
                    )
                    sleep(self.request_delay_seconds)
                else:
                    logger.error(
                        f"Failed after {self.max_retries} retries ({type(e).__name__}). "
                        f"Using default art: {card_art_uri}"
                    )
                    self._ensure_fallback_art(path)
                    return

    def card_exists_locally(self, card_id: str, cmc: int) -> bool:
        """Check if a card already exists locally without loading all IDs into memory.

        Args:
            card_id: Unique Scryfall card ID
            cmc: Converted mana cost

        Returns:
            True if the card JSON exists locally, and its art too when card art
            is enabled
        """
        if not self._get_card_path(card_id, cmc).exists():
            return False

        return not self.card_art_enabled or self._get_art_path(card_id).exists()

    def needs_refresh(self) -> bool:
        """Check if a refresh is needed by comparing metadata timestamps.

        Returns:
            True if refresh is needed, False if data is up to date
        """
        metadata = self.get_metadata()
        if not metadata or not metadata.get('updated_at'):
            logger.info("No metadata found or no timestamp. Refresh needed.")
            return True

        try:
            bulk_metadata = self.download_bulk_metadata()
            local_updated_at = metadata.get('updated_at')
            remote_updated_at = bulk_metadata.get('updated_at')

            if local_updated_at != remote_updated_at:
                logger.info(
                    f"Update available. Local: {local_updated_at}, Remote: {remote_updated_at}")
                return True

            logger.info(
                f"Card data is up to date (last updated: {local_updated_at})")
            return False
        except requests.RequestException:
            logger.warning(
                "Unable to check for updates. Assuming no refresh needed.")
            return False

    def generate_metadata(self, bulk_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Generate metadata file with card counts and update timestamp.

        Args:
            bulk_metadata: Optional bulk metadata from Scryfall
        """
        metadata_path = self.cards_path / self.METADATA_FILENAME

        metadata = {
            'updated_at': bulk_metadata.get('updated_at') if bulk_metadata else None,
            'download_uri': (bulk_metadata.get(self.BULK_DOWNLOAD_URI_FIELD)
                             if bulk_metadata else None),
            'total_card_count': self.get_total_card_count(),
            'cmc_card_count': {str(cmc): self.get_card_count_by_cmc(cmc)
                               for cmc in self.get_valid_cmcs()}
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.debug(f"Generated metadata at {metadata_path}")

    def get_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk.

        Note: This method has a side effect - it will generate a new metadata file
        if one doesn't exist. Use with caution in read-only contexts.

        Returns:
            Metadata dictionary containing:
            - updated_at: Timestamp of last database update
            - download_uri: URI used for bulk download
            - total_card_count: Total number of cards stored
            - cmc_card_count: Dictionary mapping CMC to card count
        """
        metadata_path = self.cards_path / self.METADATA_FILENAME

        if not metadata_path.exists():
            logger.warning(
                "Metadata file not found. Generating new metadata...")
            self.generate_metadata()

        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process_and_save_card(self, card: Dict[str, Any]) -> None:
        """Process and save a single card's JSON data and artwork.

        This method:
        1. Determines the card's CMC and creates the directory if needed
        2. Saves the card JSON data to the appropriate CMC folder
        3. Downloads and saves the card artwork if available and enabled

        Args:
            card: Card data dictionary from Scryfall (must contain 'id' and 'cmc')

        Raises:
            KeyError: If card is missing required fields
            OSError: If filesystem operations fail
        """
        card_id = card['id']
        cmc = int(card.get('cmc', 0))

        # Ensure CMC directory exists
        cmc_path = self.cards_path / str(cmc)
        cmc_path.mkdir(parents=True, exist_ok=True, mode=self.access_rights)

        # Save card JSON
        card_path = self._get_card_path(card_id, cmc)
        self.save_card(card_path, card)

        # Save card art if available and enabled
        if not self.card_art_enabled:
            return

        card_art_uri = self._get_card_art_uri(card)
        if card_art_uri:
            art_path = self._get_art_path(card_id)
            self.save_card_art(art_path, card_art_uri)

    def _remove_obsolete_cards(self, valid_card_ids: Set[str]) -> int:
        """Remove cards that are no longer valid in the Scryfall database.

        Args:
            valid_card_ids: Set of card IDs that are still valid

        Returns:
            Number of cards removed
        """
        logger.info("Scanning for obsolete cards to remove...")
        removed_count = 0

        for cmc_dir in self.cards_path.iterdir():
            if not (cmc_dir.is_dir() and cmc_dir.name.isdigit()):
                continue

            for card_file in cmc_dir.iterdir():
                if card_file.is_file() and card_file.suffix == self.CARD_FILE_EXTENSION:
                    card_id = card_file.stem
                    if card_id not in valid_card_ids:
                        # Remove card JSON and art
                        card_file.unlink()
                        art_file = self._get_art_path(card_id)
                        if art_file.exists():
                            art_file.unlink()
                        removed_count += 1

        if removed_count > 0:
            logger.info(f"Removed {removed_count} obsolete cards")

        return removed_count

    def _stream_and_process_cards(self, bulk_metadata: Dict[str, Any], force_full_refresh: bool) -> Dict[str, Any]:
        """Stream cards from Scryfall and process them incrementally.

        Args:
            bulk_metadata: Metadata containing download URI
            force_full_refresh: Whether to reprocess all cards

        Returns:
            Dictionary containing processing statistics:
            - total_processed: Total cards processed
            - total_creatures: Valid creature cards found
            - new_cards: Number of new cards added
            - skipped_cards: Number of existing cards skipped
            - valid_card_ids: Set of all valid card IDs

        Raises:
            requests.RequestException: If streaming fails
        """
        headers = self._get_request_headers(include_encoding=True)
        stats = {
            'total_processed': 0,
            'total_creatures': 0,
            'new_cards': 0,
            'skipped_cards': 0,
            'valid_card_ids': set()
        }

        logger.info("Streaming bulk data from Scryfall...")

        try:
            with requests.get(bulk_metadata[self.BULK_DOWNLOAD_URI_FIELD], headers=headers,
                              stream=True, timeout=self.REQUEST_TIMEOUT) as response:
                response.raise_for_status()

                # Bulk data is JSON Lines: one card object per line
                with gzip.GzipFile(fileobj=response.raw) as unzipped_stream:
                    for line in unzipped_stream:
                        line = line.strip()
                        if not line:
                            continue

                        card = json.loads(line)
                        stats['total_processed'] += 1

                        if self.is_valid_momir_basic_card(card):
                            stats['total_creatures'] += 1
                            card_id = card['id']
                            cmc = int(card.get('cmc', 0))
                            stats['valid_card_ids'].add(card_id)

                            # Check if card already exists locally
                            if force_full_refresh or not self.card_exists_locally(card_id, cmc):
                                self.process_and_save_card(card)
                                stats['new_cards'] += 1
                            else:
                                stats['skipped_cards'] += 1

                        # Progress logging
                        if stats['total_processed'] % self.PROGRESS_LOG_INTERVAL == 0:
                            self._log_progress(stats, force_full_refresh)
        except requests.RequestException as e:
            logger.error(f"Error streaming bulk data: {e}")
            raise

        return stats

    def refresh_card_data(self, force_full_refresh: bool = False) -> None:
        """Refresh card data, downloading only new/updated cards unless force_full_refresh is True.

        Args:
            force_full_refresh: If True, delete and re-download all data. 
                              If False, perform incremental update.

        Raises:
            requests.RequestException: If API requests fail
            Exception: If processing fails for any other reason
        """
        # Check if refresh is needed
        if not force_full_refresh and not self.needs_refresh():
            logger.info("No refresh needed. Card data is already up to date.")
            return

        # Ensure directories exist
        self.create_directory(self.cards_path)
        if self.card_art_enabled:
            self.create_directory(self.art_path)

        # Clear existing data if doing full refresh
        if force_full_refresh:
            logger.info(
                "Performing full refresh - all cards will be re-downloaded")
            self.delete_directory(self.cards_path)
            self.create_directory(self.cards_path)
            self.delete_directory(self.art_path)
            if self.card_art_enabled:
                self.create_directory(self.art_path)
        else:
            logger.info("Performing incremental update...")

        # Download metadata and stream cards
        try:
            bulk_metadata = self.download_bulk_metadata()
            stats = self._stream_and_process_cards(
                bulk_metadata, force_full_refresh)

            # Clean up obsolete cards (incremental update only)
            removed_cards = 0
            if not force_full_refresh:
                removed_cards = self._remove_obsolete_cards(
                    stats['valid_card_ids'])

            # Log final statistics
            if force_full_refresh:
                logger.info(f"Finished processing {stats['total_processed']} cards. "
                            f"Found {stats['total_creatures']} valid creatures.")
            else:
                logger.info(f"Finished incremental update. Processed {stats['total_processed']} cards, "
                            f"added {stats['new_cards']} new cards, removed {removed_cards} obsolete cards.")

            # Update metadata
            self.generate_metadata(bulk_metadata)

        except Exception as e:
            logger.error(f"Error during card refresh: {e}")
            raise
