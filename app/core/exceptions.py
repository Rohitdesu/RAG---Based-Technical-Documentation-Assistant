class ApplicationError(Exception):
    """Base application exception."""


class ConfigurationError(ApplicationError):
    """Raised when required configuration is missing."""


class CorpusNotReadyError(ApplicationError):
    """Raised when the vector store does not contain any documents."""


class IngestionError(ApplicationError):
    """Raised when documents cannot be ingested."""


class WebSearchError(ApplicationError):
    """Raised when the Tavily fallback search cannot return usable results."""


class SessionNotFoundError(ApplicationError):
    """Raised when a requested session does not exist."""


class UpstreamServiceError(ApplicationError):
    """Raised when an external model or API is temporarily unavailable."""
