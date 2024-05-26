from typing import Literal, Any
import datetime


def iso_timestamp_now() -> str:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    return now.isoformat()


def log(
    msg: str,
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
    file: Any | None = None,
    flush: Literal[False] = False,
):
    print(
        f"[{iso_timestamp_now()}] {msg}",
        *values,
        sep=sep,
        end=end,
        file=file,
        flush=flush,
    )
