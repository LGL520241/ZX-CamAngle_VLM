from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


class HikvisionError(RuntimeError):
    pass


class HikvisionClient:
    """海康云眸开放平台客户端，只封装本项目需要的接口。"""

    def __init__(self, base_url: str, client_id: str, client_secret: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: Optional[str] = None
        self._token_expire_at: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not self.configured:
            raise HikvisionError("未配置 client_id / client_secret")
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            if path != "/oauth/token":
                token = await self.get_token()
                headers = {"Authorization": f"bearer {token}"}
            else:
                headers = {}
            resp = await client.request(
                method, url, params=params, json=json, data=data, headers=headers
            )
        if resp.status_code == 401 and path != "/oauth/token":
            self._token = None
            raise HikvisionError("access_token 已失效，需要重新获取")
        if resp.status_code >= 400:
            raise HikvisionError(f"接口请求失败 HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            payload = resp.json()
        except Exception:
            return {"code": resp.status_code, "message": resp.text[:200]}
        code = payload.get("code")
        if code is not None and str(code) not in {"200", "0"}:
            raise HikvisionError(f"接口业务错误 code={code}: {payload.get('message')}")
        return payload

    async def get_token(self) -> str:
        if self._token and time.time() < self._token_expire_at - 60:
            return self._token
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }
        data = await self._request("POST", "/oauth/token", data=payload)
        token = data.get("access_token")
        if not token:
            raise HikvisionError(f"获取 access_token 失败: {data}")
        self._token = token
        self._token_expire_at = time.time() + int(data.get("expires_in", 604800))
        return token

    async def list_store_cameras(self, page_no: int = 1, page_size: int = 999) -> Dict[str, Any]:
        """云眸连锁：查询所有门店下的设备通道列表。"""
        return await self._request(
            "GET",
            "/v1/customization/stores/cameraList",
            params={"pageNo": page_no, "pageSize": page_size},
        )

    async def capture(self, device_serial: str, channel_no: int, quality: int = 0) -> str:
        data = await self._request(
            "POST",
            "/v1/customization/capture",
            params={"deviceSerial": device_serial, "channelNo": channel_no},
        )
        pic_url = (data.get("data") or {}).get("picUrl")
        if not pic_url:
            raise HikvisionError(f"抓图失败: {data}")
        return pic_url

    async def download(self, url: str, dest: Path) -> Path:
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    async def discover_cameras(
        self, group_nos: Optional[List[str]] = None, region_filter: str = ""
    ) -> List[Dict[str, Any]]:
        """遍历云眸连锁下的设备通道，只保留启用的通道。"""
        targets: List[Dict[str, Any]] = []
        page_no = 1
        page_size = 999
        while True:
            data = await self.list_store_cameras(page_no, page_size)
            payload = data.get("data") or {}
            rows = payload.get("rows") or []
            for ch in rows:
                serial = ch.get("deviceSerial")
                channel_no = ch.get("channelNo")
                if not serial or channel_no is None:
                    continue
                if str(ch.get("isUse", 1)) not in {"1", "true", "True"}:
                    continue
                store_name = ch.get("storeName", "") or ""
                if region_filter and region_filter not in store_name:
                    continue
                targets.append(
                    {
                        "device_serial": serial,
                        "channel_no": int(channel_no),
                        "channel_name": ch.get("channelName", ""),
                        "group_no": ch.get("storeId", ""),
                        "store_id": ch.get("storeId", ""),
                        "store_name": store_name,
                        "store_no": ch.get("storeNo", ""),
                    }
                )
            if not payload.get("hasNextPage", False) or page_no * page_size >= int(payload.get("total", 0)):
                break
            page_no += 1
        return targets
