from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from .config import settings


class EmailCryptoNotConfigured(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    if not settings.EMAIL_ENCRYPTION_KEY:
        raise EmailCryptoNotConfigured(
            "EMAIL_ENCRYPTION_KEY is not set - generate one with "
            "`python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"` and add it to .env."
        )
    return Fernet(settings.EMAIL_ENCRYPTION_KEY.encode())


def encrypt_secret(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as e:
        raise EmailCryptoNotConfigured(
            "Stored SMTP password can't be decrypted with the current "
            "EMAIL_ENCRYPTION_KEY - it was likely encrypted with a different "
            "key. Re-enter the SMTP password in the admin UI."
        ) from e
