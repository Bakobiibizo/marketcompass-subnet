import os
from fastapi import FastAPI
import requests
from loguru import logger
from communex.module import Module, endpoint
from communex.key import generate_keypair
from keylimiter import TokenBucketLimiter
from substrateinterface.keypair import Keypair


class Miner(Module):
    def __init__(self) -> None:
        super().__init__()
        self.bearer_token: str | None = os.getenv("MC_BEARER_TOKEN")

    @endpoint
    def generate(
        self,
        prompt: str,
        start_time: str = "2024-04-01T5:00:00Z",
        max_results: int = 50,
    ):
        try:
            # TODO: pass start_time, max_results from validator
            url = "https://api.twitter.com/2/tweets/search/all"

            def bearer_oauth(r):
                r.headers["Authorization"] = f"Bearer {self.bearer_token}"
                r.headers["User-Agent"] = "v2FullArchiveSearchPython"
                return r

            response: requests.Response = requests.request(
                method="GET",
                url=url,
                auth=bearer_oauth,
                params={
                    "query": prompt,
                    "max_results": max_results,
                    "start_time": start_time,
                    "user.fields": "id,username,name",
                    "tweet.fields": "created_at,author_id",
                },
                timeout=30
            )

            if response.ok:
                tweets = response.json()
                return tweets["data"]
        except Exception as e:
            logger.error(f"Error getting tweets: {e}")
        return []


if __name__ == "__main__":
    from communex.module.server import ModuleServer
    import uvicorn

    key: Keypair = generate_keypair()
    miner = Miner()
    refill_rate = 1 / 400
    bucket = TokenBucketLimiter(bucket_size=2, refill_rate=refill_rate)
    server = ModuleServer(module=miner, key=key, ip_limiter=bucket, subnets_whitelist=[17])
    app: FastAPI = server.get_fastapi_app()

    # Only allow local connections
    uvicorn.run(app, host="0.0.0.0", port=8000)
