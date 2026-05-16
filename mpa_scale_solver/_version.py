"""Single source of truth for the package version.

Lives in its own module so provenance / validation / sidecar can read it
without circular-importing the package root.
"""

__version__ = "1.0.0"
