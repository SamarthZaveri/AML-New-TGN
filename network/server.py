"""Serve the offline suspicious-network frontend."""

from __future__ import annotations

import argparse
import json
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

SAFE_NETWORK_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class NetworkStore:
    def __init__(self, data_dir: Path) -> None:
        required = [
            "networks.parquet",
            "network_nodes.parquet",
            "network_edges.parquet",
            "flow_paths.parquet",
        ]
        missing = [name for name in required if not (data_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing network output: {', '.join(missing)}")

        self.networks = pd.read_parquet(data_dir / "networks.parquet")
        self.nodes = pd.read_parquet(data_dir / "network_nodes.parquet")
        self.edges = pd.read_parquet(data_dir / "network_edges.parquet")
        self.paths = pd.read_parquet(data_dir / "flow_paths.parquet")

    @staticmethod
    def records(frame: pd.DataFrame) -> list[dict]:
        return json.loads(frame.to_json(orient="records"))

    def list_networks(self, limit: int) -> list[dict]:
        columns = [
            "component_id", "network_score", "network_bin", "account_count",
            "red_accounts", "yellow_accounts", "transaction_count",
            "suspicious_value", "flow_path_count",
        ]
        columns = [name for name in columns if name in self.networks]
        return self.records(
            self.networks.sort_values("network_score", ascending=False)
            .head(limit)[columns]
        )

    def detail(self, network_id: str) -> dict:
        if not SAFE_NETWORK_ID.fullmatch(network_id):
            raise KeyError(network_id)

        summary = self.networks.loc[self.networks.component_id == network_id]
        if summary.empty:
            raise KeyError(network_id)

        nodes = self.nodes.loc[self.nodes.component_id == network_id].copy()
        edges = self.edges.loc[self.edges.component_id == network_id].copy()
        paths = self.paths.loc[self.paths.component_id == network_id].copy()

        nodes = nodes.rename(columns={"node_id": "account_id"})
        nodes = nodes.sort_values("risk_score", ascending=False)
        edges = edges.sort_values("edge_risk_score", ascending=False).head(1000)
        paths = paths.sort_values("path_risk_score", ascending=False).head(250)

        return {
            "network": self.records(summary)[0],
            "nodes": self.records(nodes),
            "edges": self.records(edges),
            "paths": self.records(paths),
        }


def handler_factory(store: NetworkStore, static_dir: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_dir), **kwargs)

        def _json(self, body, status=HTTPStatus.OK):
            encoded = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/health":
                    return self._json({"ok": True, "network_count": len(store.networks)})

                if parsed.path == "/api/networks":
                    limit = int(parse_qs(parsed.query).get("limit", [100])[0])
                    return self._json(
                        {"networks": store.list_networks(max(1, min(limit, 500)))}
                    )

                if parsed.path.startswith("/api/network/"):
                    network_id = parsed.path.rsplit("/", 1)[-1]
                    return self._json(store.detail(network_id))
            except (KeyError, ValueError) as error:
                return self._json(
                    {"error": str(error) or "Unknown network"},
                    HTTPStatus.NOT_FOUND,
                )

            return super().do_GET()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    store = NetworkStore(args.data_dir)
    static_dir = Path(__file__).parent / "static"
    server = ThreadingHTTPServer(
        (args.host, args.port),
        handler_factory(store, static_dir),
    )

    print(f"Network UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nNetwork UI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()