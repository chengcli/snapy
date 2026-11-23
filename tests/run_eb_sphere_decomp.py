"""Compare a two-process cubed-sphere transport step with its serial reference."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


def main():
    command = [
        str(Path(sys.argv[1]).resolve()),
        "--gtest_filter=SphereEmbeddedTransport.owner_graph_agrees_after_panel_subdivision",
    ]
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ, BACKEND="gloo", DEVICE="cpu", WORLD_SIZE="1",
                   RANK="0", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1",
                   SNAPY_SPHERE_REFERENCE=str(Path(directory) / "reference.pt"))
        for _, interface in socket.if_nameindex():
            if interface in ("lo", "lo0"):
                env.setdefault("GLOO_SOCKET_IFNAME", interface)
                break
        subprocess.run(command, env=env, check=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = str(sock.getsockname()[1])
        children = []
        try:
            for rank in range(2):
                rank_env = dict(env, WORLD_SIZE="2", RANK=str(rank),
                                LOCAL_RANK=str(rank), MASTER_PORT=port)
                children.append(subprocess.Popen(command, env=rank_env))
            for child in children:
                if child.wait(timeout=90):
                    raise RuntimeError("distributed cubed-sphere comparison failed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait()


if __name__ == "__main__":
    main()
