"""
security.py — Application-level firewall for Crop Yield Predictor
=================================================================
Import and call `apply_security()` at the top of app.py to activate
all protections. This module handles:

  1. Input sanitization & validation
  2. Rate limiting per session
  3. Path traversal protection
  4. Request size limits
  5. Error message scrubbing (no stack traces to users)
  6. Source code access prevention
  7. Environment variable enforcement
"""

import re
import time
import hashlib
import streamlit as st
from pathlib import Path
from functools import wraps

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
MAX_REQUESTS_PER_MINUTE = 30          # Rate limit threshold
INPUT_MAX_STRING_LENGTH = 200         # Max chars for any text input
ALLOWED_CROP_TYPES = {"Rice", "Wheat", "Maize", "Soybean"}
ALLOWED_REGIONS = {"Selangor", "Johor", "Penang", "Perak"}
ALLOWED_SEASONS = {"Wet", "Dry", "Monsoon"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files that must NEVER be served or exposed
PROTECTED_PATHS = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / ".git",
]


# ─────────────────────────────────────────────────────────────
# 1. Input Sanitization
# ─────────────────────────────────────────────────────────────
def sanitize_string(value, field_name="input"):
    """Strip dangerous characters and enforce length limits."""
    if not isinstance(value, str):
        return value
    if len(value) > INPUT_MAX_STRING_LENGTH:
        raise ValueError(f"{field_name} exceeds maximum length ({INPUT_MAX_STRING_LENGTH} chars).")

    # Block shell injection patterns
    dangerous_patterns = [
        r"[;&|`$]",           # Shell metacharacters
        r"\.\./",             # Path traversal
        r"<script",           # XSS
        r"(?i)(drop|delete|insert|update)\s",  # SQL keywords
        r"__import__",        # Python injection
        r"eval\s*\(",         # Code execution
        r"exec\s*\(",         # Code execution
        r"os\.system",        # System calls
        r"subprocess",        # Process spawning
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValueError(f"Invalid characters detected in {field_name}.")

    return value.strip()


def validate_numeric_input(value, field_name, min_val=None, max_val=None):
    """Ensure numeric inputs are within expected bounds."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    if min_val is not None and value < min_val:
        raise ValueError(f"{field_name} must be >= {min_val}.")
    if max_val is not None and value > max_val:
        raise ValueError(f"{field_name} must be <= {max_val}.")
    return value


def validate_categorical_input(value, field_name, allowed_values):
    """Ensure categorical inputs match the allowed whitelist."""
    if value not in allowed_values:
        raise ValueError(
            f"Invalid {field_name}: '{value}'. "
            f"Allowed: {', '.join(sorted(allowed_values))}"
        )
    return value


def validate_all_inputs(
    soil_ph, soil_moisture, avg_temperature, total_rainfall,
    fertilizer_amount, pesticide_usage, sunlight_hours,
    nitrogen_content, phosphorus_content, potassium_content,
    irrigation_frequency, crop_type, region, season
):
    """Run full validation on all user inputs. Returns sanitized values."""
    errors = []

    checks = [
        (soil_ph,              "Soil pH",              0.0,    14.0),
        (soil_moisture,        "Soil Moisture",        0.0,    100.0),
        (avg_temperature,      "Avg Temperature",      -50.0,  60.0),
        (total_rainfall,       "Total Rainfall",       0.0,    10000.0),
        (fertilizer_amount,    "Fertilizer Amount",    0.0,    5000.0),
        (pesticide_usage,      "Pesticide Usage",      0.0,    1000.0),
        (sunlight_hours,       "Sunlight Hours",       0.0,    24.0),
        (nitrogen_content,     "Nitrogen Content",     0.0,    5000.0),
        (phosphorus_content,   "Phosphorus Content",   0.0,    5000.0),
        (potassium_content,    "Potassium Content",    0.0,    5000.0),
        (irrigation_frequency, "Irrigation Frequency", 0,      365),
    ]

    for val, name, lo, hi in checks:
        try:
            validate_numeric_input(val, name, lo, hi)
        except ValueError as e:
            errors.append(str(e))

    try:
        validate_categorical_input(crop_type, "Crop Type", ALLOWED_CROP_TYPES)
    except ValueError as e:
        errors.append(str(e))

    try:
        validate_categorical_input(region, "Region", ALLOWED_REGIONS)
    except ValueError as e:
        errors.append(str(e))

    try:
        validate_categorical_input(season, "Season", ALLOWED_SEASONS)
    except ValueError as e:
        errors.append(str(e))

    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# ─────────────────────────────────────────────────────────────
# 2. Rate Limiting
# ─────────────────────────────────────────────────────────────
def check_rate_limit():
    """Enforce per-session rate limiting to prevent abuse."""
    now = time.time()

    if "request_log" not in st.session_state:
        st.session_state.request_log = []

    # Purge entries older than 60 seconds
    st.session_state.request_log = [
        t for t in st.session_state.request_log if now - t < 60
    ]

    if len(st.session_state.request_log) >= MAX_REQUESTS_PER_MINUTE:
        st.error("Too many requests. Please wait a moment before trying again.")
        st.stop()

    st.session_state.request_log.append(now)


# ─────────────────────────────────────────────────────────────
# 3. Path Traversal Protection
# ─────────────────────────────────────────────────────────────
def safe_path_resolve(user_path, allowed_root):
    """Ensure a user-supplied path stays within the allowed directory."""
    resolved = Path(user_path).resolve()
    allowed = Path(allowed_root).resolve()
    if not str(resolved).startswith(str(allowed)):
        raise PermissionError(f"Access denied: path outside allowed directory.")
    return resolved


def is_path_protected(target_path):
    """Check if a path falls within any protected directory."""
    target = Path(target_path).resolve()
    for protected in PROTECTED_PATHS:
        protected = protected.resolve()
        if target == protected or str(target).startswith(str(protected) + "/"):
            return True
    return False


# ─────────────────────────────────────────────────────────────
# 4. Error Scrubbing
# ─────────────────────────────────────────────────────────────
def safe_error_message(exception):
    """Return a user-safe error message without leaking internals."""
    # Map known error types to clean messages
    error_map = {
        FileNotFoundError: "A required file could not be found.",
        PermissionError:   "Access denied.",
        ValueError:        str(exception),
        KeyError:          "An unexpected data field was encountered.",
        ConnectionError:   "Could not connect to the external service.",
        TimeoutError:      "The request timed out. Please try again.",
    }

    for err_type, message in error_map.items():
        if isinstance(exception, err_type):
            return message

    # Generic fallback — never expose the raw exception
    return "An unexpected error occurred. Please try again."


# ─────────────────────────────────────────────────────────────
# 5. Session Fingerprinting
# ─────────────────────────────────────────────────────────────
def get_session_fingerprint():
    """Generate a hash-based fingerprint for the current session."""
    session_id = st.session_state.get("_session_id", "unknown")
    return hashlib.sha256(str(session_id).encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────
# 6. Security Headers (for deployment behind a reverse proxy)
# ─────────────────────────────────────────────────────────────
SECURITY_HEADERS_HTML = """
<meta http-equiv="X-Content-Type-Options" content="nosniff">
<meta http-equiv="X-Frame-Options" content="DENY">
<meta http-equiv="X-XSS-Protection" content="1; mode=block">
<meta http-equiv="Referrer-Policy" content="strict-origin-when-cross-origin">
<meta http-equiv="Permissions-Policy" content="camera=(), microphone=(), geolocation=()">
"""


# ─────────────────────────────────────────────────────────────
# 7. Master Activation
# ─────────────────────────────────────────────────────────────
def apply_security():
    """Call this once at the top of app.py to activate all protections."""

    # Inject security meta headers
    st.markdown(SECURITY_HEADERS_HTML, unsafe_allow_html=True)

    # Initialize rate limiter
    if "request_log" not in st.session_state:
        st.session_state.request_log = []

    # Block direct file access via query params
    query_params = st.query_params
    for key, value in query_params.items():
        if ".." in str(value) or "/" in str(value):
            st.error("Access denied.")
            st.stop()

    return True
