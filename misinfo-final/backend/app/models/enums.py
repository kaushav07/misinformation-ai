from enum import Enum


class Verdict(str, Enum):
    TRUE = "TRUE"
    FAKE = "FAKE"
    MISLEADING = "MISLEADING"
    UNVERIFIED = "UNVERIFIED"
    SATIRE = "SATIRE"


class Language(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    UNKNOWN = "unknown"


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    URL = "url"


class RiskLevel(str, Enum):
    LOW = "LOW"          # 0-30
    MEDIUM = "MEDIUM"    # 31-60
    HIGH = "HIGH"        # 61-80
    CRITICAL = "CRITICAL"  # 81-100
