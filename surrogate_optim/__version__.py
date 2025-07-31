"""Version information for surrogate-gradient-optim-lab."""

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

# Build information (updated during CI/CD)
__build_date__ = "2024-07-30"
__git_commit__ = "unknown"
__git_branch__ = "main"

# Package metadata
__title__ = "surrogate-gradient-optim-lab"
__description__ = "Toolkit for offline black-box optimization using learned gradient surrogates"
__author__ = "Daniel Schmidt"
__author_email__ = "daniel@terragon-labs.com"
__url__ = "https://github.com/terragon-labs/surrogate-gradient-optim-lab"
__license__ = "MIT"

def get_version():
    """Return the version string."""
    return __version__

def get_version_info():
    """Return version information as a dictionary."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "build_date": __build_date__,
        "git_commit": __git_commit__,
        "git_branch": __git_branch__,
        "title": __title__,
        "description": __description__,
        "author": __author__,
        "author_email": __author_email__,
        "url": __url__,
        "license": __license__,
    }