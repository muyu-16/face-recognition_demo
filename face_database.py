# -*- coding: utf-8 -*-
"""
face_database.py
================
人脸库（Gallery）管理模块：增、删、查 + 本地持久化。

持久化方案：pickle
    人脸库本质上就是"一堆矩阵 + 一点元数据"：
        {姓名: {'embeddings': ndarray(N,512), 'image_paths': [...], ...}}
    这种结构用 pickle 直接序列化最省事，读回来还是原样的 numpy 数组，
    不需要像 JSON 那样把浮点数组转成列表（又慢又占空间）。

磁盘上的文件布局：:

    项目根目录/
    ├── face_db.pkl              # 人脸库主文件（pickle：姓名 + 特征向量 + 元数据）
    ├── face_db.bak.pkl          # 上一次保存的自动备份（覆盖前生成）
    └── registered_faces/        # 注册时裁剪下来的人脸小图（供界面缩略图展示）
        ├── 张三/
        │   ├── 0000.png
        │   └── 0001.png
        └── 李四/
            └── 0000.png

安全提示：pickle 反序列化会执行文件里的任意代码，所以只加载自己生成的人脸库文件，
不要把别人给的 .pkl 当成数据文件随便打开（README 里有同样说明）。
"""

from __future__ import annotations

import os
import pickle
import re
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SCHEMA_VERSION = 1
DEFAULT_DB_PATH = Path("face_db.pkl")
DEFAULT_FACE_DIR = Path("registered_faces")

# Windows / Linux 文件名非法字符
_ILLEGAL_NAME_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def sanitize_name(name: str) -> str:
    """清洗姓名：去掉首尾空白与非法文件名字符，避免生成出错的目录名。"""
    cleaned = _ILLEGAL_NAME_CHARS.sub("_", str(name)).strip().strip(".")
    return cleaned


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class PersonInfo:
    """一个人的注册信息（界面展示用）。"""

    name: str
    num_images: int
    created_at: str
    updated_at: str
    image_paths: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "姓名": self.name,
            "图片数": self.num_images,
            "注册时间": self.created_at,
            "更新时间": self.updated_at,
        }


class FaceDatabase:
    """人脸库：负责特征的增删查与本地 pickle 持久化。"""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        face_dir: Path | str = DEFAULT_FACE_DIR,
        auto_load: bool = True,
        save_face_images: bool = True,
        l2_normalize: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.face_dir = Path(face_dir)
        self.save_face_images = bool(save_face_images)
        self.l2_normalize = bool(l2_normalize)

        self.people: Dict[str, dict] = {}      # 姓名 -> 该人的注册数据
        self.meta: Dict[str, object] = {}      # 骨干网络、特征维度、更新时间等
        self.last_error: Optional[str] = None  # 加载失败时的提示，界面可展示

        if auto_load:
            self.load()

    # ==================================================================
    # 持久化
    # ==================================================================
    def _empty_payload(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": _now(),
            "updated_at": _now(),
            "meta": {},
            "people": {},
        }

    def load(self, force: bool = False) -> "FaceDatabase":
        """从 pickle 文件读取人脸库。

        - 文件不存在：初始化空库（首次运行属于正常情况）；
        - 文件损坏：把坏文件改名备份，然后以空库启动，并在 last_error 里说明原因。
        """
        if not self.db_path.exists():
            payload = self._empty_payload()
            self.people = payload["people"]
            self.meta = payload["meta"]
            self.created_at = payload["created_at"]
            self.updated_at = payload["updated_at"]
            return self

        if self.people and not force:
            return self  # 已经加载过就不重复读盘

        try:
            with open(self.db_path, "rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict) or "people" not in payload:
                raise ValueError("人脸库文件结构不符合预期")
            self.people = payload.get("people", {}) or {}
            self.meta = payload.get("meta", {}) or {}
            self.created_at = payload.get("created_at", _now())
            self.updated_at = payload.get("updated_at", _now())
            self.last_error = None
        except Exception as exc:
            broken = self.db_path.with_suffix(f".corrupt-{int(time.time())}.pkl")
            try:
                shutil.move(str(self.db_path), str(broken))
                detail = f"原文件已改名为 {broken.name}"
            except Exception:
                detail = "无法备份损坏文件"
            self.last_error = f"人脸库文件读取失败（{exc}），{detail}，已按空库启动。"
            payload = self._empty_payload()
            self.people = payload["people"]
            self.meta = payload["meta"]
            self.created_at = payload["created_at"]
            self.updated_at = payload["updated_at"]
        return self

    def save(self) -> Path:
        """原子写盘：先写临时文件再替换，避免中途断电导致人脸库损坏。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()

        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": getattr(self, "created_at", _now()),
            "updated_at": self.updated_at,
            "meta": self.meta,
            "people": self.people,
        }

        # 覆盖前先备份上一版
        if self.db_path.exists():
            try:
                shutil.copy2(self.db_path, self.db_path.with_name(self.db_path.stem + ".bak.pkl"))
            except Exception:
                pass  # 备份失败不影响主流程

        tmp_path = self.db_path.with_name(self.db_path.name + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.db_path)   # 同一文件系统内替换是原子操作
        return self.db_path

    def export_bytes(self) -> bytes:
        """把当前人脸库序列化成字节，供 Streamlit 的下载按钮使用。"""
        return pickle.dumps(
            {"schema_version": SCHEMA_VERSION, "created_at": getattr(self, "created_at", _now()),
             "updated_at": self.updated_at, "meta": self.meta, "people": self.people},
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # ==================================================================
    # 增
    # ==================================================================
    def add_person(
        self,
        name: str,
        embeddings: np.ndarray,
        face_images: Optional[Sequence[np.ndarray]] = None,
        overwrite: bool = False,
        backbone: Optional[str] = None,
        append: bool = True,
    ) -> PersonInfo:
        """注册一个人（或给已注册的人追加样本）。

        参数：
            name        : 姓名（会做非法字符清洗）
            embeddings  : (N, D) 特征矩阵，N 张图，D 维（本项目 512）
            face_images : 对应的裁剪人脸图（uint8 RGB），用于存缩略图，可为 None
            overwrite   : True=先把该人原有样本清空再写入；False=追加
            backbone    : 记录用的骨干网络名，便于界面上判断是否需要重新注册
            append      : False 时若该人已存在则报错（严格模式）
        """
        safe_name = sanitize_name(name)
        if not safe_name:
            raise ValueError("姓名不能为空（或只包含非法字符）")

        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.size == 0:
            raise ValueError("特征为空，请至少提供一张包含人脸的照片")
        if not np.all(np.isfinite(emb)):
            raise ValueError("特征向量中出现 NaN/Inf，请检查输入图片")

        # 特征是否 L2 归一化：统一在入库时做一次，保证库内、查询时口径一致
        if self.l2_normalize:
            emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)

        # 维度一致性检查：换了骨干网络后维度或特征空间会变，必须重新注册
        existing_dim = self.embedding_dim
        if existing_dim is not None and existing_dim != emb.shape[1]:
            raise ValueError(
                f"特征维度不一致：人脸库现有 {existing_dim} 维，本次 {emb.shape[1]} 维。"
                f"更换骨干网络后请先清空人脸库再重新注册。"
            )

        if safe_name in self.people and not overwrite and not append:
            raise ValueError(f"『{safe_name}』已存在；如需覆盖请勾选覆盖选项。")

        created_at = self.people.get(safe_name, {}).get("created_at", _now())
        old = self.people.get(safe_name, {})
        old_emb = np.asarray(old.get("embeddings", np.zeros((0, emb.shape[1]), np.float32)))
        old_paths: List[str] = list(old.get("image_paths", []))

        if overwrite:
            new_emb = emb
            new_paths: List[str] = []
            start_index = 0
        else:
            new_emb = np.concatenate([old_emb, emb], axis=0) if old_emb.size else emb
            new_paths = old_paths
            start_index = len(old_paths)

        # 保存裁剪后的人脸小图，便于在界面里展示"这个人注册了哪些脸"。
        # 注意：存图只是"可视化辅助"，即使失败也不能影响特征入库这个核心功能，
        # 所以这里把异常兜住并记录到 meta 里，由界面提示。
        if face_images is not None and self.save_face_images:
            try:
                person_dir = self.face_dir / safe_name
                if overwrite and person_dir.exists():
                    shutil.rmtree(person_dir, ignore_errors=True)
                person_dir.mkdir(parents=True, exist_ok=True)

                saved_paths: List[str] = []
                for offset, img in enumerate(face_images):
                    file_path = person_dir / f"{start_index + offset:04d}.png"
                    arr = np.asarray(img, dtype=np.uint8)
                    Image.fromarray(arr).save(file_path)
                    saved_paths.append(str(file_path))
                new_paths = new_paths + saved_paths
                self.meta.pop("image_save_error", None)
            except Exception as exc:
                self.meta["image_save_error"] = (
                    f"人脸小图保存失败（特征已正常入库，只是界面里看不到缩略图）：{exc}"
                )

        self.people[safe_name] = {
            "embeddings": np.ascontiguousarray(new_emb, dtype=np.float32),
            "image_paths": new_paths,
            "created_at": created_at,
            "updated_at": _now(),
            "backbone": backbone or self.meta.get("backbone"),
        }
        self.meta["backbone"] = backbone or self.meta.get("backbone")
        self.meta["embedding_dim"] = int(new_emb.shape[1])
        self.meta["num_people"] = len(self.people)
        self.meta["num_embeddings"] = int(sum(len(v["embeddings"]) for v in self.people.values()))
        self.save()

        return PersonInfo(
            name=safe_name,
            num_images=int(new_emb.shape[0]),
            created_at=created_at,
            updated_at=self.people[safe_name]["updated_at"],
            image_paths=new_paths,
        )

    # ==================================================================
    # 删
    # ==================================================================
    def delete_person(self, name: str) -> bool:
        """删除指定人员的全部特征，并清理其人脸小图目录。返回是否真的删掉了。"""
        safe_name = sanitize_name(name)
        if safe_name not in self.people:
            return False

        del self.people[safe_name]
        person_dir = self.face_dir / safe_name
        if person_dir.exists():
            shutil.rmtree(person_dir, ignore_errors=True)

        self.meta["num_people"] = len(self.people)
        self.meta["num_embeddings"] = int(sum(len(v["embeddings"]) for v in self.people.values()))
        self.save()
        return True

    def rename_person(self, old_name: str, new_name: str) -> bool:
        """给已注册人员改名（特征不变，只改索引与图片目录）。"""
        old_safe, new_safe = sanitize_name(old_name), sanitize_name(new_name)
        if old_safe not in self.people or not new_safe:
            return False
        if new_safe in self.people and new_safe != old_safe:
            raise ValueError(f"『{new_safe}』已存在，无法重名。")

        record = self.people.pop(old_safe)
        # 同步重命名图片目录
        old_dir, new_dir = self.face_dir / old_safe, self.face_dir / new_safe
        if old_dir.exists():
            if new_dir.exists():
                shutil.rmtree(new_dir, ignore_errors=True)
            old_dir.rename(new_dir)
            record["image_paths"] = [str(new_dir / Path(p).name) if p else "" for p in record.get("image_paths", [])]
        record["updated_at"] = _now()
        self.people[new_safe] = record
        self.save()
        return True

    def clear(self) -> None:
        """清空整个人脸库（特征 + 图片 + 磁盘文件）。"""
        self.people = {}
        if self.face_dir.exists():
            shutil.rmtree(self.face_dir, ignore_errors=True)
        self.meta["num_people"] = 0
        self.meta["num_embeddings"] = 0
        self.save()

    # ==================================================================
    # 查
    # ==================================================================
    @property
    def embedding_dim(self) -> Optional[int]:
        """人脸库当前使用的特征维度；库为空时返回 None。"""
        for record in self.people.values():
            emb = np.asarray(record.get("embeddings"))
            if emb.size:
                return int(emb.shape[1])
        return None

    @property
    def backbone(self) -> Optional[str]:
        return self.meta.get("backbone")

    def has_person(self, name: str) -> bool:
        return sanitize_name(name) in self.people

    def get_person(self, name: str) -> Optional[dict]:
        return self.people.get(sanitize_name(name))

    def get_embeddings(self, name: str) -> Optional[np.ndarray]:
        record = self.get_person(name)
        return None if record is None else np.asarray(record["embeddings"], dtype=np.float32)

    def list_people(self) -> List[PersonInfo]:
        """列出所有已注册人员（按注册时间排序），界面表格直接用这个。"""
        infos = [
            PersonInfo(
                name=name,
                num_images=int(len(record.get("embeddings", []))),
                created_at=record.get("created_at", ""),
                updated_at=record.get("updated_at", ""),
                image_paths=[p for p in record.get("image_paths", []) if p],
            )
            for name, record in self.people.items()
        ]
        infos.sort(key=lambda info: (info.created_at, info.name))
        return infos

    def names(self) -> List[str]:
        """所有已注册姓名。"""
        return [info.name for info in self.list_people()]

    def get_gallery(self) -> Tuple[List[str], np.ndarray]:
        """导出检索用的"扁平人脸库"。

        返回：
            names   : 长度 N 的姓名列表，同一个人的多张注册图会重复出现
            gallery : (N, D) 特征矩阵
        identify() 就吃这两个东西。
        """
        names: List[str] = []
        feats: List[np.ndarray] = []
        for name in self.names():
            emb = self.get_embeddings(name)
            if emb is None or emb.size == 0:
                continue
            for row in emb:
                names.append(name)
                feats.append(row)
        if not feats:
            dim = self.embedding_dim or 512
            return [], np.zeros((0, dim), dtype=np.float32)
        return names, np.stack(feats, axis=0).astype(np.float32)

    @property
    def num_people(self) -> int:
        return len(self.people)

    @property
    def num_embeddings(self) -> int:
        return int(sum(len(np.asarray(v.get("embeddings", []))) for v in self.people.values()))

    def stats(self) -> Dict[str, object]:
        """人脸库概况，界面顶部指标卡用。"""
        return {
            "注册人数": self.num_people,
            "人脸特征总数": self.num_embeddings,
            "特征维度": self.embedding_dim,
            "骨干网络": self.backbone,
            "数据库文件": str(self.db_path),
            "文件大小(KB)": round(self.db_path.stat().st_size / 1024, 1) if self.db_path.exists() else 0.0,
            "最后更新": self.updated_at,
        }

    # ==================================================================
    # 其他
    # ==================================================================
    def set_meta(self, key: str, value: object) -> None:
        self.meta[key] = value

    def get_meta(self, key: str, default: object = None) -> object:
        return self.meta.get(key, default)

    def clone(self) -> "FaceDatabase":
        return deepcopy(self)

    def __len__(self) -> int:
        return self.num_people

    def __contains__(self, name: object) -> bool:
        return sanitize_name(str(name)) in self.people

    def __repr__(self) -> str:
        return (f"FaceDatabase(people={self.num_people}, embeddings={self.num_embeddings}, "
                f"dim={self.embedding_dim}, file='{self.db_path}')")


# ---------------------------------------------------------------------------
# 模块自测：python face_database.py
# 会在 .selftest/ 目录里建一个临时库，验证增删查与读写往返，跑完自动清理
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import shutil

    work = Path(".selftest")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    db = FaceDatabase(db_path=work / "test_db.pkl", face_dir=work / "faces")
    print("初始状态：", db)

    rng = np.random.default_rng(0)
    for name, count in [("张三", 5), ("李四", 5)]:
        base = rng.normal(size=512).astype(np.float32)
        embs = np.stack([base + 0.1 * rng.normal(size=512).astype(np.float32) for _ in range(count)])
        fake_imgs = [rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8) for _ in range(count)]
        info = db.add_person(name, embs, face_images=fake_imgs, backbone="inception_resnet_v1")
        print("注册：", info.as_dict())

    names, gallery = db.get_gallery()
    print(f"扁平人脸库：{len(names)} 条特征，矩阵形状 {gallery.shape}")
    print("概况：", db.stats())

    # 重新从磁盘加载，验证 pickle 往返是否保住 numpy 数组
    db2 = FaceDatabase(db_path=work / "test_db.pkl", face_dir=work / "faces")
    print("重新加载后：", db2.stats())
    assert np.allclose(db2.get_embeddings("张三"), db.get_embeddings("张三")), "pickle 往返不一致！"

    print("删除李四：", db2.delete_person("李四"))
    print("剩余人员：", db2.names())
    print("删除不存在的人：", db2.delete_person("不存在"))

    print("模块自测通过 ✅")
    db2.clear()
    shutil.rmtree(work, ignore_errors=True)
