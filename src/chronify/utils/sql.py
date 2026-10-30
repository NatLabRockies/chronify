from uuid import uuid4


TEMP_TABLE_PREFIX = "chronify"


def make_temp_view_name(prefix: str = TEMP_TABLE_PREFIX) -> str:
    """Make a random name to be used as a temporary view or table."""
    return f"{prefix}_{uuid4().hex}"
