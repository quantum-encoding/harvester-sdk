"""
Harvester SDK - Complete AI Processing Platform

Commercial interface to the enterprise Harvesting Engine.
Delegates all operations to the proven, battle-tested core components.

Copyright (c) 2025 Quantum Encoding Ltd.
Licensed under the Harvester Commercial License.
"""

__version__ = "2.0.0"
__author__ = "Quantum Encoding Ltd."
__license__ = "Commercial"

# Harvester SDK - imports from the proven engine
from .sdk import (
    HarvesterSDK,
    HarvesterClient,
    HarvesterLicense,
    quick_process,
    quick_council
)

# Primary SDK exports
__all__ = [
    "HarvesterSDK",
    "HarvesterClient", 
    "HarvesterLicense",
    "quick_process",
    "quick_council"
]

# Version info for license validation
def get_version_info():
    """Get version information for license validation"""
    return {
        "version": __version__,
        "author": __author__, 
        "license": __license__,
        "engine": "enterprise_harvesting_engine",
        "wrapper": "harvester_sdk_v1"
    }