#!/usr/bin/env python3
"""Download and cache all CARI aesthetic images.

This script downloads all ~2,811 images from the CARI dataset and caches them
locally for future use. Images will be saved to data/cari_aesthetics/images/
and metadata will be cached in data/cari_aesthetics/documents_cache.json.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multiview.docsets.cari_aesthetics import CARIAestheticsDocSet

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    """Download all CARI images and cache them."""
    logger.info("Starting CARI image download...")
    logger.info("This will download ~2,811 images and may take 10-20 minutes.")

    # Create docset without max_docs limit to download everything
    docset = CARIAestheticsDocSet(
        criterion="aesthetic_name",
        config={},  # No max_docs = download all
    )

    try:
        # This will download and cache all images
        documents = docset.load_documents()

        logger.info(f"✓ Successfully downloaded and cached {len(documents)} images!")
        logger.info("Images saved to: data/cari_aesthetics/images/")
        logger.info("Cache saved to: data/cari_aesthetics/documents_cache.json")

        # Print some stats
        aesthetics = {}
        for doc in documents:
            aesthetic = doc.get("aesthetic_name", "Unknown")
            aesthetics[aesthetic] = aesthetics.get(aesthetic, 0) + 1

        logger.info(f"Downloaded images from {len(aesthetics)} aesthetic movements:")
        for aesthetic, count in sorted(
            aesthetics.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            logger.info(f"  {aesthetic}: {count} images")

        if len(aesthetics) > 10:
            logger.info(f"  ... and {len(aesthetics) - 10} more aesthetics")

        return 0

    except Exception as e:
        logger.error(f"Failed to download images: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
