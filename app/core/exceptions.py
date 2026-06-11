class ApplicationError(Exception):
    """Base application exception."""


class ConfigurationError(ApplicationError):
    """Raised when required configuration is missing."""


class CorpusNotReadyError(ApplicationError):
    """Raised when the vector store does not contain any documents."""


class IngestionError(ApplicationError):
    """Raised when documents cannot be ingested."""
