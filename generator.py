import secrets

def generate_session_id(prefix: str = "sess", entropy_bytes: int = 24) -> str:
    """
    Generate a strong, URL-safe session_id.
    
    - Uses `secrets` (CSPRNG).
    - `entropy_bytes=24` ≈ 192 bits of entropy -> ~32 chars token.
    - URL-safe (base64url), good for headers, filenames, DB keys.
    """
    if entropy_bytes < 16:
        raise ValueError("entropy_bytes should be at least 16 for strong IDs.")
    token = secrets.token_urlsafe(entropy_bytes).rstrip("=")  # trim padding if any
    return f"{prefix}-{token}"
