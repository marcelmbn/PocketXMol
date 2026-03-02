import gc
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from Bio import PDB
from Bio.PDB import PDBIO
from Bio.SeqUtils import seq1
from easydict import EasyDict
from rdkit import Chem
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

sys.path.append(".")

from models.maskfill import PMAsymDenoiser
from models.sample import get_cfd_traj, sample_loop3, seperate_outputs2
from process.utils_process import (
    add_pep_bb_data,
    extract_pocket,
    get_input_from_file,
    get_peptide_info,
    make_dummy_mol_with_coordinate,
)
from scripts.train_pl import DataModule
from utils.dataset import UseDataset
from utils.misc import (
    CaptureLogger,
    get_logger,
    get_new_log_dir,
    make_config,
    save_config,
    seed_all,
)
from utils.reconstruct import (
    MolReconsError,
    create_sdf_string,
    reconstruct_from_generated_with_edges,
    reconstruct_pdb_from_generated,
)
from utils.sample_noise import get_sample_noiser
from utils.transforms import Compose, get_transforms


@dataclass
class SamplingRequest:
    protein_path: Optional[str] = None
    pocket_coord: Optional[Sequence[float]] = None
    radius: Optional[float] = None
    outdir: str = "./outputs_use"
    device: str = "cuda:0"
    num_mols: Optional[int] = None
    batch_size: Optional[int] = None
    num_workers: int = -1
    shuffle: bool = False
    seed: Optional[int] = None
    save_traj_prob: Optional[float] = None
    num_steps: Optional[int] = None
    max_ar_step: Optional[int] = None
    num_repeats: Optional[int] = None
    data_id: Optional[str] = None
    pdbid: Optional[str] = None
    config_task: str = "configs/sample/examples/sbdd.yml"
    config_model: str = "configs/sample/pxm.yml"
    checkpoint: Optional[str] = None
    pocket_center: Optional[Sequence[float]] = None
    variable_mol_size: Optional[Dict[str, Any]] = None
    save_output: Optional[Sequence[str]] = None


def print_pool_status(pool, logger, is_pep: bool = False) -> None:
    if not is_pep:
        logger.info(
            "[Pool] Succ/Incomp/Bad: %d/%d/%d"
            % (len(pool.succ), len(pool.incomp), len(pool.bad))
        )
    else:
        logger.info(
            "[Pool] Succ/Nonstd/Incomp/Bad: %d/%d/%d/%d"
            % (len(pool.succ), len(pool.nonstd), len(pool.incomp), len(pool.bad))
        )


def get_input_data(
    protein_path,
    input_ligand=None,
    is_pep=False,
    pocket_args={},
    pocmol_args={},
):
    ref_ligand = pocket_args.get("ref_ligand_path", None)
    pocket_coord = pocket_args.get("pocket_coord", None)
    if ref_ligand is not None:
        pass
    elif pocket_coord is not None:
        ref_ligand = make_dummy_mol_with_coordinate(pocket_coord)
    else:
        print(
            "Neither ref_ligand nor pocket_coord provided for pocket extraction. "
            "Using input_ligand as reference."
        )
        assert input_ligand is not None and (
            input_ligand.endswith(".sdf") or input_ligand.endswith(".pdb")
        ), "Only SDF/PDB input_ligand can be used for pocket extraction."
        ref_ligand = input_ligand

    pocket_pdb = extract_pocket(
        protein_path,
        ref_ligand,
        radius=pocket_args.get("radius", 10),
        criterion=pocket_args.get("criterion", "center_of_mass"),
    )
    pocmol_data, mol = get_input_from_file(
        input_ligand, pocket_pdb, return_mol=True, **pocmol_args
    )

    if is_pep:
        if input_ligand.endswith(".pdb"):
            pep_info = get_peptide_info(input_ligand)
            assert torch.isclose(
                pocmol_data["pos_all_confs"][0], pep_info["peptide_pos"], 1e-2
            ).all(), "Molecule and peptide atoms may not match"
        elif input_ligand.startswith("peplen_"):
            pep_info = add_pep_bb_data(pocmol_data)
        else:
            pep_info = {}
        pocmol_data.update(pep_info)

    return pocmol_data, pocket_pdb, mol


def _build_config(request: SamplingRequest) -> EasyDict:
    config = make_config(request.config_task, request.config_model)
    if request.num_mols is not None:
        config.sample.num_mols = request.num_mols
    if request.protein_path is not None:
        config.data.protein_path = request.protein_path
    config.data.pocket_args = EasyDict(config.data.get("pocket_args", {}))
    if request.pocket_coord is not None:
        config.data.pocket_args.pocket_coord = list(request.pocket_coord)
    if request.radius is not None:
        config.data.pocket_args.radius = request.radius
    config.data.pocmol_args = EasyDict(config.data.get("pocmol_args", {}))
    if request.data_id is not None:
        config.data.pocmol_args.data_id = request.data_id
    if request.pdbid is not None:
        config.data.pocmol_args.pdbid = request.pdbid

    config.transforms = EasyDict(config.get("transforms", {}))
    config.transforms.featurizer_pocket = EasyDict(
        config.transforms.get("featurizer_pocket", {})
    )
    pocket_center = request.pocket_center or request.pocket_coord
    if pocket_center is not None:
        config.transforms.featurizer_pocket.center = list(pocket_center)

    if request.variable_mol_size is not None:
        config.transforms.variable_mol_size = EasyDict(request.variable_mol_size)
    if request.seed is not None:
        config.sample.seed = request.seed
    if request.save_traj_prob is not None:
        config.sample.save_traj_prob = request.save_traj_prob
    if request.num_steps is not None:
        config.noise.num_steps = request.num_steps
    if request.max_ar_step is not None and "ar_config" in config.noise:
        config.noise.ar_config.max_ar_step = request.max_ar_step
    if request.num_repeats is not None:
        config.sample.num_repeats = request.num_repeats
    if request.checkpoint is not None:
        config.model.checkpoint = request.checkpoint
    if request.save_output is not None:
        config.sample.save_output = list(request.save_output)
    return config


def _get_logger(name: str, log_dir: str):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger = get_logger(name, log_dir)
    logger.propagate = False
    return logger


def _merge_request_kwargs(
    request: Optional[SamplingRequest],
    overrides: Dict[str, Any],
) -> SamplingRequest:
    clean_overrides = {
        key: value for key, value in overrides.items() if value is not None
    }
    if request is None:
        return SamplingRequest(**clean_overrides)
    merged = asdict(request)
    merged.update(clean_overrides)
    return SamplingRequest(**merged)


def run_sampling(
    protein_path: Optional[str] = None,
    pocket_coord: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
    *,
    outdir: Optional[str] = None,
    device: Optional[str] = None,
    num_mols: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
    seed: Optional[int] = None,
    save_traj_prob: Optional[float] = None,
    num_steps: Optional[int] = None,
    max_ar_step: Optional[int] = None,
    num_repeats: Optional[int] = None,
    data_id: Optional[str] = None,
    pdbid: Optional[str] = None,
    config_task: Optional[str] = None,
    config_model: Optional[str] = None,
    checkpoint: Optional[str] = None,
    pocket_center: Optional[Sequence[float]] = None,
    variable_mol_size: Optional[Dict[str, Any]] = None,
    save_output: Optional[Sequence[str]] = None,
    request: Optional[SamplingRequest] = None,
) -> Dict[str, Any]:
    """Run PocketXMol sampling from Python.

    You can call this either with direct keyword arguments:

    `run_sampling(protein_path=..., pocket_coord=..., radius=..., num_mols=...)`

    or by passing a `SamplingRequest`:

    `run_sampling(request=SamplingRequest(...))`
    """
    if isinstance(protein_path, SamplingRequest):
        if request is not None:
            raise ValueError(
                "Pass either a positional SamplingRequest or request=..., not both."
            )
        request = protein_path
        protein_path = None
    request = _merge_request_kwargs(
        request,
        {
            "protein_path": protein_path,
            "pocket_coord": pocket_coord,
            "radius": radius,
            "outdir": outdir,
            "device": device,
            "num_mols": num_mols,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": shuffle,
            "seed": seed,
            "save_traj_prob": save_traj_prob,
            "num_steps": num_steps,
            "max_ar_step": max_ar_step,
            "num_repeats": num_repeats,
            "data_id": data_id,
            "pdbid": pdbid,
            "config_task": config_task,
            "config_model": config_model,
            "checkpoint": checkpoint,
            "pocket_center": pocket_center,
            "variable_mol_size": variable_mol_size,
            "save_output": save_output,
        },
    )
    config = _build_config(request)
    config_name = os.path.basename(request.config_task).replace(".yml", "")
    if request.config_model is not None:
        config_name += "_" + os.path.basename(request.config_model).replace(".yml", "")

    seed = config.sample.seed + np.sum(
        [ord(s) for s in request.outdir] + [ord(s) for s in request.config_task]
    )
    seed_all(seed)
    config.sample.complete_seed = seed.item()

    ckpt = torch.load(
        config.model.checkpoint, map_location=request.device, weights_only=False
    )
    cfg_dir = os.path.dirname(config.model.checkpoint).replace(
        "checkpoints", "train_config"
    )
    train_config = make_config(os.path.join(cfg_dir, "".join(os.listdir(cfg_dir))))

    save_traj_prob = config.sample.save_traj_prob
    batch_size = (
        config.sample.batch_size if request.batch_size is None else request.batch_size
    )
    num_mols = config.sample.get("num_mols", 100)
    num_repeats = config.sample.get("num_repeats", 1)

    log_root = request.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    logger = _get_logger("sample_api", log_dir)
    logger.info("Load from %s..." % config.model.checkpoint)
    logger.info({"request": asdict(request)})
    logger.info(config)
    save_config(config, os.path.join(log_dir, os.path.basename(request.config_task)))

    sdf_dir = os.path.join(log_dir, "SDF")
    pure_sdf_dir = os.path.join(log_dir, os.path.basename(log_dir) + "_SDF")
    os.makedirs(sdf_dir, exist_ok=True)
    os.makedirs(pure_sdf_dir, exist_ok=True)
    df_path = os.path.join(log_dir, "gen_info.csv")

    logger.info("Loading data placeholder...")
    for samp_trans in config.get("transforms", {}).keys():
        if samp_trans in train_config.transforms.keys():
            train_config.transforms.get(samp_trans).update(
                config.transforms.get(samp_trans)
            )
    dm = DataModule(train_config)
    featurizer_list = dm.get_featurizers()
    featurizer = featurizer_list[-1]
    in_dims = dm.get_in_dims()
    task_trans = get_transforms(config.task.transform, mode="use")
    is_ar = config.task.transform.get("name", "")
    noiser = get_sample_noiser(
        config.noise,
        in_dims["num_node_types"],
        in_dims["num_edge_types"],
        mode="sample",
        device=request.device,
        ref_config=train_config.noise,
    )

    if "variable_mol_size" in getattr(config, "transforms", []):
        transforms = featurizer_list + [
            get_transforms(config.transforms.variable_mol_size),
            task_trans,
        ]
    elif "variable_sc_size" in getattr(config, "transforms", []):
        transforms = featurizer_list + [
            get_transforms(config.transforms.variable_sc_size),
            task_trans,
        ]
    else:
        transforms = featurizer_list + [task_trans]

    addition_transforms = [
        get_transforms(tr) for tr in config.data.get("transforms", [])
    ]
    transforms = Compose(transforms + addition_transforms)
    follow_batch = sum(
        [getattr(t, "follow_batch", []) for t in transforms.transforms], []
    )
    exclude_keys = sum(
        [getattr(t, "exclude_keys", []) for t in transforms.transforms], []
    )

    logger.info("Loading dataset...")
    data_cfg = config.data
    is_pep = data_cfg.get("is_pep", None)
    if is_pep is None:
        input_ligand = data_cfg.get("input_ligand", "")
        is_pep = input_ligand.endswith(".pdb") or input_ligand.startswith("pep")
    data, pocket_block, in_mol = get_input_data(
        protein_path=data_cfg.protein_path,
        input_ligand=data_cfg.get("input_ligand", None),
        is_pep=is_pep,
        pocket_args=data_cfg.get("pocket_args", {}),
        pocmol_args=data_cfg.get("pocmol_args", {}),
    )
    test_set = UseDataset(
        data, n=num_mols, task=config.task.name, transforms=transforms
    )
    test_loader = DataLoader(
        test_set,
        batch_size,
        shuffle=request.shuffle,
        num_workers=train_config.train.num_workers
        if request.num_workers == -1
        else request.num_workers,
        pin_memory=train_config.train.pin_memory,
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
    )

    input_pocmol_dir = os.path.join(pure_sdf_dir, "0_inputs")
    os.makedirs(input_pocmol_dir, exist_ok=True)
    pocket_block_path = os.path.join(input_pocmol_dir, "pocket_block.pdb")
    input_mol_path = os.path.join(input_pocmol_dir, "input_mol.sdf")
    with open(pocket_block_path, "w") as f:
        f.write(pocket_block)
    Chem.MolToMolFile(in_mol, input_mol_path)

    logger.info("Loading diffusion model...")
    if train_config.model.name == "pm_asym_denoiser":
        model = PMAsymDenoiser(config=train_config.model, **in_dims).to(request.device)
    else:
        raise NotImplementedError(f"Unsupported model: {train_config.model.name}")
    model.load_state_dict(
        {
            k[6:]: value
            for k, value in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    )
    model.eval()

    pool = EasyDict(
        {"succ": [], "bad": [], "incomp": [], **({"nonstd": []} if is_pep else {})}
    )
    info_keys = ["data_id", "db", "task", "key"]
    i_saved = 0
    result_rows: List[Dict[str, Any]] = []
    result_mols: List[Dict[str, Any]] = []
    logger.info("Start sampling... (Total: n_mols=%d)" % num_mols)

    try:
        for i_repeat in range(num_repeats):
            logger.info("Generating molecules.")
            for batch in test_loader:
                if i_saved >= num_mols:
                    logger.info("Enough molecules. Stop sampling.")
                    break

                batch = batch.to(request.device)
                batch, outputs, trajs = sample_loop3(
                    batch, model, noiser, request.device, is_ar=is_ar
                )
                data_list = [
                    {key: batch[key][i] for key in info_keys} for i in range(len(batch))
                ]
                generated_list, outputs_list, traj_list_dict = seperate_outputs2(
                    batch, outputs, trajs
                )

                mol_info_list = []
                for i_mol in tqdm(
                    range(len(generated_list)), desc="Post process generated mols"
                ):
                    mol_info = featurizer.decode_output(**generated_list[i_mol])
                    mol_info.update(data_list[i_mol])

                    try:
                        if not is_pep:
                            with CaptureLogger():
                                rdmol = reconstruct_from_generated_with_edges(
                                    mol_info, in_mol=in_mol
                                )
                            smiles = Chem.MolToSmiles(rdmol)
                            if "." in smiles:
                                tag = "incomp"
                                pool.incomp.append(mol_info)
                                logger.warning("Incomplete molecule: %s" % smiles)
                            else:
                                tag = ""
                                pool.succ.append(mol_info)
                                logger.info("Success: %s" % smiles)
                        else:
                            with CaptureLogger():
                                pdb_struc, rdmol = reconstruct_pdb_from_generated(
                                    mol_info, gt_path=data_cfg.input_ligand
                                )
                            aaseq = seq1(
                                "".join(res.resname for res in pdb_struc.get_residues())
                            )
                            if rdmol is None:
                                rdmol = Chem.MolFromSmiles("")
                            smiles = Chem.MolToSmiles(rdmol)
                            if "." in smiles:
                                tag = "incomp"
                                pool.incomp.append(mol_info)
                                logger.warning("Incomplete molecule: %s" % aaseq)
                            elif "X" in aaseq:
                                tag = "nonstd"
                                pool.nonstd.append(mol_info)
                                logger.warning("Non-standard amino acid: %s" % aaseq)
                            else:
                                tag = ""
                                pool.succ.append(mol_info)
                                logger.info("Success: %s" % aaseq)
                    except MolReconsError:
                        pool.bad.append(mol_info)
                        logger.warning("Reconstruction error encountered.")
                        smiles = ""
                        tag = "bad"
                        rdmol = create_sdf_string(mol_info)
                        if is_pep:
                            aaseq = ""
                            pdb_struc = PDB.Structure.Structure("bad")

                    mol_info.update(
                        {
                            "rdmol": rdmol,
                            "smiles": smiles,
                            "tag": tag,
                            "output": outputs_list[i_mol],
                            **(
                                {
                                    "pdb_struc": pdb_struc,
                                    "aaseq": aaseq,
                                }
                                if is_pep
                                else {}
                            ),
                        }
                    )

                    if np.random.rand() < save_traj_prob:
                        mol_traj = {}
                        for traj_who in traj_list_dict.keys():
                            traj_this_mol = traj_list_dict[traj_who][i_mol]
                            for t in range(len(traj_this_mol["node"])):
                                mol_this = featurizer.decode_output(
                                    node=traj_this_mol["node"][t],
                                    pos=traj_this_mol["pos"][t],
                                    halfedge=traj_this_mol["halfedge"][t],
                                    halfedge_index=generated_list[i_mol][
                                        "halfedge_index"
                                    ],
                                    pocket_center=generated_list[i_mol][
                                        "pocket_center"
                                    ],
                                )
                                mol_this = create_sdf_string(mol_this)
                                mol_traj.setdefault(traj_who, []).append(mol_this)
                        mol_info["traj"] = mol_traj
                    mol_info_list.append(mol_info)

                df_info_batch = []
                for data_finished in mol_info_list:
                    rdmol = data_finished["rdmol"]
                    tag = data_finished["tag"]
                    filename_base = str(i_saved) + (f"-{tag}" if tag else "")
                    if is_pep:
                        pdb_struc = data_finished["pdb_struc"]
                        filename_pdb = filename_base + ".pdb"
                        pdb_io = PDBIO()
                        pdb_io.set_structure(pdb_struc)
                        pdb_io.save(os.path.join(pure_sdf_dir, filename_pdb))
                    filename_sdf = filename_base + (
                        ".sdf" if not is_pep else "_mol.sdf"
                    )
                    sdf_path = os.path.join(pure_sdf_dir, filename_sdf)
                    if tag != "bad":
                        Chem.MolToMolFile(rdmol, sdf_path)
                    else:
                        with open(sdf_path, "w+") as f:
                            f.write(rdmol)
                    if "traj" in data_finished:
                        for traj_who in data_finished["traj"].keys():
                            sdf_file = "$$$$\n".join(data_finished["traj"][traj_who])
                            name_traj = filename_base + f"-{traj_who}.sdf"
                            with open(os.path.join(sdf_dir, name_traj), "w+") as f:
                                f.write(sdf_file)
                    i_saved += 1

                    output = data_finished["output"]
                    cfd_traj = get_cfd_traj(output["confidence_pos_traj"])
                    cfd_pos = output["confidence_pos"].detach().cpu().numpy().mean()
                    cfd_node = output["confidence_node"].detach().cpu().numpy().mean()
                    cfd_edge = (
                        output["confidence_halfedge"].detach().cpu().numpy().mean()
                    )
                    save_output = getattr(config.sample, "save_output", [])
                    if len(save_output) > 0:
                        output_to_save = {key: output[key] for key in save_output}
                        torch.save(
                            output_to_save, os.path.join(sdf_dir, filename_base + ".pt")
                        )

                    info_dict = {
                        key: data_finished[key]
                        for key in info_keys
                        + (["aaseq"] if is_pep else [])
                        + ["smiles", "tag"]
                    }
                    info_dict.update(
                        {
                            "filename": filename_sdf if not is_pep else filename_pdb,
                            "i_repeat": i_repeat,
                            "cfd_traj": cfd_traj,
                            "cfd_pos": cfd_pos,
                            "cfd_node": cfd_node,
                            "cfd_edge": cfd_edge,
                        }
                    )
                    df_info_batch.append(info_dict)
                    result_rows.append(info_dict)
                    result_mols.append(
                        {
                            "smiles": data_finished["smiles"],
                            "tag": tag,
                            "filename": filename_sdf if not is_pep else filename_pdb,
                            "sdf_path": sdf_path,
                            "pdb_path": os.path.join(pure_sdf_dir, filename_pdb)
                            if is_pep
                            else None,
                            "rdmol": None if tag == "bad" else rdmol,
                            "cfd_traj": cfd_traj,
                            "cfd_pos": cfd_pos,
                            "cfd_node": cfd_node,
                            "cfd_edge": cfd_edge,
                        }
                    )

                df_info_batch = pd.DataFrame(df_info_batch)
                if os.path.exists(df_path):
                    df_info = pd.read_csv(df_path)
                    df_info = pd.concat([df_info, df_info_batch], ignore_index=True)
                else:
                    df_info = df_info_batch
                df_info.to_csv(df_path, index=False)
                print_pool_status(pool, logger, is_pep=is_pep)

                del batch, outputs, trajs, mol_info_list[0 : len(mol_info_list)]
                if request.device != "cpu":
                    with torch.cuda.device(request.device):
                        torch.cuda.empty_cache()
                gc.collect()

        dummy_pool = {key: [""] * len(value) for key, value in pool.items()}
        torch.save(dummy_pool, os.path.join(log_dir, "samples_all.pt"))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt. Stop sampling.")

    return {
        "log_dir": log_dir,
        "sdf_dir": pure_sdf_dir,
        "traj_dir": sdf_dir,
        "df_path": df_path,
        "pocket_block_path": pocket_block_path,
        "input_mol_path": input_mol_path,
        "summary": {
            "succ": len(pool.succ),
            "incomp": len(pool.incomp),
            "bad": len(pool.bad),
            **({"nonstd": len(pool.nonstd)} if is_pep else {}),
        },
        "molecules": result_mols,
        "records": result_rows,
        "config": config,
    }


def run_sampling_simple(
    protein_path: str,
    pocket_coord: Sequence[float],
    radius: float,
    *,
    num_mols: int = 100,
    batch_size: Optional[int] = None,
    num_steps: Optional[int] = None,
    outdir: str = "./outputs_use",
    device: str = "cuda:0",
    data_id: Optional[str] = None,
    pdbid: Optional[str] = None,
) -> Dict[str, Any]:
    """Run sampling with only the most relevant inputs.

    This wrapper keeps the rest of the behavior on the example-config defaults.
    """
    return run_sampling(
        protein_path=protein_path,
        pocket_coord=pocket_coord,
        radius=radius,
        num_mols=num_mols,
        batch_size=batch_size,
        num_steps=num_steps,
        outdir=outdir,
        device=device,
        data_id=data_id,
        pdbid=pdbid,
    )
