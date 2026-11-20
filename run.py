import socket

import uvicorn

from cam_angle.config import get_config


def find_free_port(start: int = 20350, end: int = 20399) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"{start}-{end} 端口全部被占用")


if __name__ == "__main__":
    cfg = get_config()
    port = find_free_port(20350, 20399)
    print(f"CamAngle 服务启动： http://127.0.0.1:{port}")
    uvicorn.run(
        "cam_angle.main:app",
        host=cfg.host,
        port=port,
        reload=False,
    )
