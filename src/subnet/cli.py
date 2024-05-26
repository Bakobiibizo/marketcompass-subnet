from substrateinterface.keypair import Keypair
import typer
from typing import Annotated

from subnet.validator._config import ValidatorSettings
from subnet.validator.validator import TwitterValidator

from communex._common import get_node_url, get  # type: ignore
from communex.client import CommuneClient  # type: ignore
from communex.compat.key import classic_load_key  # type: ignore


app = typer.Typer()


@app.command(name="serve-subnet")
def serve(
    commune_key: Annotated[
        str, typer.Argument(help="Name of the key present in `~/.commune/key`")
    ],
    call_timeout: int = 30,
):
    keypair: Keypair = classic_load_key(name=commune_key)  # type: ignore
    settings = ValidatorSettings()  # type: ignore
    c_client = CommuneClient(url=get_node_url())
    validator = TwitterValidator(
        key=keypair,
        netuid=settings.subnetuid,
        client=c_client,
        call_timeout=call_timeout,
    )
    validator.validation_loop(settings=settings)


if __name__ == "__main__":
    typer.run(function=serve)
