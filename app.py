from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pocketxmol import run_sampling, run_sampling_simple

app = FastAPI(title="PocketXMol Service", version="1.0")


class SamplingSimpleRequest(BaseModel):
    protein_path: str = Field(..., description="Path to the protein PDB file.")
    pocket_coord: List[float] = Field(
        ..., min_length=3, max_length=3, description="Pocket center [x, y, z]."
    )
    radius: float = Field(
        ..., gt=0, description="Pocket extraction radius in Angstrom."
    )
    num_mols: int = Field(default=100, ge=1, le=1000)
    batch_size: Optional[int] = Field(default=None, ge=1, le=1000)
    num_steps: Optional[int] = Field(default=None, ge=1, le=1000)
    outdir: str = Field(default="./outputs_api")
    device: str = Field(default="cuda:0")
    data_id: Optional[str] = Field(default=None)
    pdbid: Optional[str] = Field(default=None)


class SamplingRequestModel(BaseModel):
    protein_path: Optional[str] = Field(default=None)
    pocket_coord: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    radius: Optional[float] = Field(default=None, gt=0)
    outdir: Optional[str] = Field(default=None)
    device: Optional[str] = Field(default=None)
    num_mols: Optional[int] = Field(default=None, ge=1, le=5000)
    batch_size: Optional[int] = Field(default=None, ge=1, le=5000)
    num_workers: Optional[int] = Field(default=None, ge=-1, le=64)
    shuffle: Optional[bool] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    save_traj_prob: Optional[float] = Field(default=None, ge=0, le=1)
    num_steps: Optional[int] = Field(default=None, ge=1, le=5000)
    max_ar_step: Optional[int] = Field(default=None, ge=0, le=5000)
    num_repeats: Optional[int] = Field(default=None, ge=1, le=100)
    data_id: Optional[str] = Field(default=None)
    pdbid: Optional[str] = Field(default=None)
    config_task: Optional[str] = Field(default=None)
    config_model: Optional[str] = Field(default=None)
    checkpoint: Optional[str] = Field(default=None)
    pocket_center: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    variable_mol_size: Optional[Dict[str, Any]] = Field(default=None)
    save_output: Optional[List[str]] = Field(default=None)


def _require_cuda(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but no CUDA GPU is available.")


def _ensure_paths_exist(
    payload: Union[SamplingRequestModel, SamplingSimpleRequest],
) -> None:
    if (
        hasattr(payload, "protein_path")
        and payload.protein_path
        and not os.path.exists(payload.protein_path)
    ):
        raise FileNotFoundError(f"Protein file not found: {payload.protein_path}")


def _json_safe_result(result: Dict[str, Any]) -> Dict[str, Any]:
    def _to_builtin(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: _to_builtin(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_builtin(val) for val in value]
        return value

    return {
        "log_dir": result["log_dir"],
        "sdf_dir": result["sdf_dir"],
        "traj_dir": result["traj_dir"],
        "df_path": result["df_path"],
        "pocket_block_path": result["pocket_block_path"],
        "input_mol_path": result["input_mol_path"],
        "summary": result["summary"],
        "molecule_count": len(result["molecules"]),
        "molecules": [
            {
                "smiles": mol["smiles"],
                "tag": mol["tag"],
                "filename": mol["filename"],
                "sdf_path": mol["sdf_path"],
                "pdb_path": mol["pdb_path"],
                "cfd_traj": _to_builtin(mol["cfd_traj"]),
                "cfd_pos": _to_builtin(mol["cfd_pos"]),
                "cfd_node": _to_builtin(mol["cfd_node"]),
                "cfd_edge": _to_builtin(mol["cfd_edge"]),
            }
            for mol in result["molecules"]
        ],
    }


def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/sample/simple")
async def sample_simple(payload: SamplingSimpleRequest) -> Dict[str, Any]:
    try:
        _require_cuda(payload.device)
        _ensure_paths_exist(payload)
        start = time.perf_counter()
        result = run_sampling_simple(
            protein_path=payload.protein_path,
            pocket_coord=payload.pocket_coord,
            radius=payload.radius,
            num_mols=payload.num_mols,
            batch_size=payload.batch_size,
            num_steps=payload.num_steps,
            outdir=payload.outdir,
            device=payload.device,
            data_id=payload.data_id,
            pdbid=payload.pdbid,
        )
        runtime = time.perf_counter() - start
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sampling failed: {exc}") from exc

    response = _json_safe_result(result)
    response["runtime_seconds"] = round(runtime, 3)
    response["mode"] = "simple"
    return response


@app.post("/sample")
async def sample(payload: SamplingRequestModel) -> Dict[str, Any]:
    try:
        device = payload.device or "cuda:0"
        _require_cuda(device)
        _ensure_paths_exist(payload)
        start = time.perf_counter()
        result = run_sampling(**_model_to_dict(payload))
        runtime = time.perf_counter() - start
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sampling failed: {exc}") from exc

    response = _json_safe_result(result)
    response["runtime_seconds"] = round(runtime, 3)
    response["mode"] = "full"
    return response
