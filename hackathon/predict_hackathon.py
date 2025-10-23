# predict_hackathon.py
import argparse
import json
import os
import shutil
import subprocess
import sys

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict

import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule

# Add torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add src to path for importing verifiers
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from boltz.model.verifiers.openmm_energy import OpenMMInteractionEnergy
from boltz.model.verifiers.rosetta_energy import RosettaInteractionEnergy
from boltz.model.verifiers.interface_cdr_pct import InterfaceCDRPct

# --- Robust Biopython Import ---
try:
    from boltz.model.verifiers.interface_cdr_pct import BIOPYTHON_AVAILABLE
except ImportError:
    BIOPYTHON_AVAILABLE = False

if not BIOPYTHON_AVAILABLE:
    print("Initial Biopython import failed. Attempting to add site-packages to path...")
    # Try to find the site-packages directory of the current environment
    # and add it to the path. This can resolve issues where the environment
    # is not fully activated in the context of the script execution.
    import site
    for sp in site.getsitepackages():
        if "site-packages" in sp:
            print(f"Adding {sp} to sys.path")
            sys.path.append(sp)
            break
    try:
        from boltz.model.verifiers.interface_cdr_pct import BIOPYTHON_AVAILABLE
        if BIOPYTHON_AVAILABLE:
            print("Successfully imported Biopython after path correction.")
    except ImportError:
        print("Failed to import Biopython even after path correction.")
        BIOPYTHON_AVAILABLE = False
# -----------------------------

from openmm.unit import kilojoules_per_mole

# ---------------------------------------------------------------------------
# ---- Helper functions for multi-metric ranking -----------------------------
# ---------------------------------------------------------------------------

def _load_confidence_scores(pdb_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Load iptm and iptm_boltz1 scores from confidence JSON file.
    
    Args:
        pdb_path: Path to the PDB file
        
    Returns:
        Tuple of (iptm_score, iptm_boltz1_score) - either can be None if not found
    """
    print(f"  [Confidence] Loading confidence scores for {pdb_path.name}")
    
    try:
        # Construct confidence file path
        pdb_name = pdb_path.stem  # e.g., "8CYH_config_3_model_0"
        confidence_file = pdb_path.parent / f"confidence_{pdb_name}.json"
        
        print(f"  [Confidence] Looking for confidence file: {confidence_file}")
        
        if not confidence_file.exists():
            print(f"  [Confidence] WARNING: Confidence file not found")
            return None, None
            
        print(f"  [Confidence] Confidence file found, loading...")
        with open(confidence_file, 'r') as f:
            confidence_data = json.load(f)
            
        iptm_score = confidence_data.get('iptm')
        iptm_boltz1_score = confidence_data.get('iptm_boltz1')
        
        print(f"  [Confidence] iptm score: {iptm_score}")
        print(f"  [Confidence] iptm_boltz1 score: {iptm_boltz1_score}")
        
        return iptm_score, iptm_boltz1_score
        
    except Exception as e:
        print(f"  [Confidence] ERROR: Could not load confidence scores for {pdb_path.name}")
        print(f"  [Confidence] Exception type: {type(e).__name__}")
        print(f"  [Confidence] Exception message: {str(e)}")
        return None, None

def _calculate_cdr_metrics(pdb_path: Path) -> Optional[Dict[str, float]]:
    """
    Calculate CDR interface metrics.
    
    Args:
        pdb_path: Path to the PDB file
        
    Returns:
        A dictionary with 'total_frac' and 'heavy_frac' on a 0-1 scale if successful, None otherwise
    """
    print(f"  [CDR Interface] Starting calculation for {pdb_path.name}")
    
    try:
        if not BIOPYTHON_AVAILABLE:
            print(f"  [CDR Interface] Biopython not available, skipping.")
            return None

        # Caching is disabled to show printouts every time
        # pdb_name = pdb_path.stem
        # result_file = pdb_path.parent / f"cdr_interface_{pdb_name}.json"
        # if result_file.exists():
        # ...
        
        print(f"  [CDR Interface] No cached result found, calculating...")
        calculator = InterfaceCDRPct()
        
        antibody_chain_ids = ['H', 'L'] 
        
        results = calculator.calculate_cdr_interface_percentage(
            pdb_file_path=str(pdb_path),
            antibody_chain_ids=antibody_chain_ids
        )
        
        total_pct = results.get('cdr_interface_pct')
        heavy_pct = results.get('heavy_cdr_interface_pct')
        
        print(f"  [CDR Interface] Total Interface Percentage: {total_pct}%")
        if heavy_pct is not None:
            print(f"  [CDR Interface] Heavy Chain Interface Percentage: {heavy_pct}%")
        
        # Save results is disabled as caching is disabled
        # print(f"  [CDR Interface] Saving results to: {result_file}")
        # with open(result_file, 'w') as f:
        #     json.dump(results, f, indent=2)
        # print(f"  [CDR Interface] Results saved successfully")
            
        return {
            'total_frac': total_pct / 100.0 if total_pct is not None else None,
            'heavy_frac': heavy_pct / 100.0 if heavy_pct is not None else None,
        }
        
    except Exception as e:
        print(f"  [CDR Interface] ERROR: Exception during calculation for {pdb_path.name}")
        print(f"  [CDR Interface] Exception type: {type(e).__name__}")
        print(f"  [CDR Interface] Exception message: {str(e)}")
        import traceback
        print(f"  [CDR Interface] Full traceback:")
        traceback.print_exc()
        return None

def _calculate_openmm_energy(pdb_path: Path) -> Optional[float]:
    """
    Calculate OpenMM interaction energy per residue.
    
    Args:
        pdb_path: Path to the PDB file
        
    Returns:
        interaction_energy_per_residue if successful, None otherwise
    """
    print(f"  [OpenMM] Starting energy calculation for {pdb_path.name}")
    
    try:
        # Check if OpenMM is available by trying to access the class
        try:
            # Try to access the imported class
            energy_calculator = OpenMMInteractionEnergy()
            print(f"  [OpenMM] OpenMMInteractionEnergy class available, proceeding...")
        except NameError:
            print(f"  [OpenMM] ERROR: OpenMMInteractionEnergy class not available")
            return None
            
        # Check if result already exists
        pdb_name = pdb_path.stem
        energy_file = pdb_path.parent / f"openmm_energy_{pdb_name}.json"
        
        if energy_file.exists():
            print(f"  [OpenMM] Found existing energy file: {energy_file}")
            with open(energy_file, 'r') as f:
                energy_data = json.load(f)
            result = energy_data.get('interaction_energy_per_residue_kj_mol')
            print(f"  [OpenMM] Loaded cached result: {result}")
            return result
        
        print(f"  [OpenMM] No cached result found, calculating energy...")
        print(f"  [OpenMM] PDB file path: {pdb_path}")
        print(f"  [OpenMM] PDB file exists: {pdb_path.exists()}")
        
        # Calculate energy
        print(f"  [OpenMM] Creating OpenMMInteractionEnergy calculator...")
        energy_calculator = OpenMMInteractionEnergy()
        print(f"  [OpenMM] Calculator created successfully")
        
        print(f"  [OpenMM] Calling calculate_interaction_energy with CUDA platform...")
        energy_results = energy_calculator.calculate_interaction_energy(
            pdb_file_path=str(pdb_path),
            platform_name='CUDA'  # Default to CUDA, fallback handled by OpenMM
        )
        print(f"  [OpenMM] Energy calculation completed successfully")
        print(f"  [OpenMM] Results keys: {list(energy_results.keys())}")
        
        # Extract interaction energy per residue
        interaction_energy_per_residue = energy_results['interaction_energy_per_residue'].value_in_unit(kilojoules_per_mole)
        print(f"  [OpenMM] Interaction energy per residue: {interaction_energy_per_residue} kJ/mol")
        
        # Save results
        result_data = {
            'interaction_energy_per_residue_kj_mol': interaction_energy_per_residue,
            'complex_energy_kj_mol': energy_results['complex_energy'].value_in_unit(kilojoules_per_mole),
            'interaction_energy_kj_mol': energy_results['interaction_energy'].value_in_unit(kilojoules_per_mole),
            'complex_energy_per_residue_kj_mol': energy_results['complex_energy_per_residue'].value_in_unit(kilojoules_per_mole)
        }
        
        print(f"  [OpenMM] Saving results to: {energy_file}")
        with open(energy_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"  [OpenMM] Results saved successfully")
            
        return interaction_energy_per_residue
        
    except Exception as e:
        print(f"  [OpenMM] ERROR: Exception during energy calculation for {pdb_path.name}")
        print(f"  [OpenMM] Exception type: {type(e).__name__}")
        print(f"  [OpenMM] Exception message: {str(e)}")
        import traceback
        print(f"  [OpenMM] Full traceback:")
        traceback.print_exc()
        return None

def _calculate_rosetta_energy(pdb_path: Path) -> Optional[float]:
    """
    Calculate Rosetta interaction energy per residue.
    
    Args:
        pdb_path: Path to the PDB file
        
    Returns:
        interaction_energy_per_residue if successful, None otherwise
    """
    print(f"  [Rosetta] Starting energy calculation for {pdb_path.name}")
    
    try:
        # Check if Rosetta is available by trying to access the class
        try:
            # Try to access the imported class
            energy_calculator = RosettaInteractionEnergy()
            print(f"  [Rosetta] RosettaInteractionEnergy class available, proceeding...")
        except NameError:
            print(f"  [Rosetta] ERROR: RosettaInteractionEnergy class not available")
            return None
            
        # Check if result already exists
        pdb_name = pdb_path.stem
        energy_file = pdb_path.parent / f"rosetta_energy_{pdb_name}.json"
        
        if energy_file.exists():
            print(f"  [Rosetta] Found existing energy file: {energy_file}")
            with open(energy_file, 'r') as f:
                energy_data = json.load(f)
            result = energy_data.get('interaction_energy_per_residue_reu')
            print(f"  [Rosetta] Loaded cached result: {result}")
            return result
        
        print(f"  [Rosetta] No cached result found, calculating energy...")
        print(f"  [Rosetta] PDB file path: {pdb_path}")
        print(f"  [Rosetta] PDB file exists: {pdb_path.exists()}")
        
        # Calculate energy
        print(f"  [Rosetta] Creating RosettaInteractionEnergy calculator...")
        energy_calculator = RosettaInteractionEnergy()
        print(f"  [Rosetta] Calculator created successfully")
        
        print(f"  [Rosetta] Calling calculate_interaction_energy...")
        energy_results = energy_calculator.calculate_interaction_energy(
            pdb_file_path=str(pdb_path)
        )
        print(f"  [Rosetta] Energy calculation completed successfully")
        print(f"  [Rosetta] Results keys: {list(energy_results.keys())}")
        
        # Extract interaction energy per residue
        interaction_energy_per_residue = energy_results['interaction_energy_per_residue']
        print(f"  [Rosetta] Interaction energy per residue: {interaction_energy_per_residue} REU")
        
        # Save results
        result_data = {
            'interaction_energy_per_residue_reu': interaction_energy_per_residue,
            'complex_energy_reu': energy_results['complex_energy'],
            'interaction_energy_reu': energy_results['interaction_energy'],
            'complex_energy_per_residue_reu': energy_results['complex_energy_per_residue']
        }
        
        print(f"  [Rosetta] Saving results to: {energy_file}")
        with open(energy_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"  [Rosetta] Results saved successfully")
            
        return interaction_energy_per_residue
        
    except Exception as e:
        print(f"  [Rosetta] ERROR: Exception during energy calculation for {pdb_path.name}")
        print(f"  [Rosetta] Exception type: {type(e).__name__}")
        print(f"  [Rosetta] Exception message: {str(e)}")
        import traceback
        print(f"  [Rosetta] Full traceback:")
        traceback.print_exc()
        return None

def _calculate_all_metrics(pdb_path: Path) -> Tuple[Path, float, List[str]]:
    """
    Calculate all available metrics for a single PDB file.
    
    Args:
        pdb_path: Path to the PDB file
        
    Returns:
        Tuple of (pdb_path, composite_score, metric_names)
    """
    scores = []
    metric_names = []
    
    # Load confidence scores (iptm and iptm_boltz1)
    iptm_score, iptm_boltz1_score = _load_confidence_scores(pdb_path)
    
    if iptm_score is not None:
        scores.append(iptm_score)
        metric_names.append("iptm")
    
    if iptm_boltz1_score is not None:
        scores.append(iptm_boltz1_score)
        metric_names.append("iptm_boltz1")
    
    # Calculate OpenMM energy
    openmm_energy = _calculate_openmm_energy(pdb_path)
    if openmm_energy is not None:
        scores.append(-openmm_energy)  # Negate so lower energy = higher score
        metric_names.append("openmm_energy")
    
    # Calculate Rosetta energy
    rosetta_energy = _calculate_rosetta_energy(pdb_path)
    if rosetta_energy is not None:
        scores.append(-rosetta_energy)  # Negate so lower energy = higher score
        metric_names.append("rosetta_energy")
    
    # Calculate CDR interface metrics for filtering
    cdr_metrics = _calculate_cdr_metrics(pdb_path)
    if cdr_metrics:
        total_frac = cdr_metrics.get('total_frac')
        heavy_frac = cdr_metrics.get('heavy_frac')
        
        # Filter out samples that have 0 contacts for total or heavy chain CDRs
        if total_frac is not None and total_frac == 0:
            print(f"  [Rank Filter] Total CDR interface is 0. Ranking as worst.")
            return pdb_path, -9999.0, ["cdr_interface_total_ZERO"]
        
        if heavy_frac is not None and heavy_frac == 0:
            print(f"  [Rank Filter] Heavy chain CDR interface is 0. Ranking as worst.")
            return pdb_path, -9999.0, ["cdr_interface_heavy_ZERO"]

        # The CDR metric is not added to the score, only used for filtering
        metric_names.append("cdr_interface_FILTERED")
    
    if not scores:
        composite_score = 0.0
    else:
        # Simple average of available metrics
        composite_score = sum(scores) / len(scores)
    
    return pdb_path, composite_score, metric_names

# ---------------------------------------------------------------------------
# ---- Participants should modify these four functions ----------------------
# ---------------------------------------------------------------------------

def prepare_protein_complex(datapoint_id: str, proteins: List[Protein], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein complex prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        proteins: List of protein sequences to predict as a complex
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `proteins`` will contain 3 chains
    # H,L: heavy and light chain of the Fv or Fab region
    # A: the antigen
    #
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Generate multiple configurations with different seeds (0-3)
    configs = []
    for seed in range(16):  # seeds 0, 1, 2, 3
        cli_args = [
            "--diffusion_samples", "4",
            "--max_parallel_samples", "1",
            "--seed", str(seed), 
            "--use_dropout",
            "--no-confidence-prediction",
            # "--use_boltz1_confidence_steering",
            # "--use_boltz1_trunk_features",
            # "--subsample_msa",
            # "--num_subsampled_msa", "128",
        ]
        configs.append((input_dict, cli_args))

    return configs

def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein-ligand prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligands: A list of a single small molecule ligand object 
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `protein` is a single-chain target protein sequence with id A
    # `ligands` contains a single small molecule ligand object with unknown binding sites
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    cli_args = ["--diffusion_samples", "5"]
    return [(input_dict, cli_args)]

def post_process_protein_complex(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein complex submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    for prediction_dir in prediction_dirs:
        config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        all_pdbs.extend(config_pdbs)

    # Sort all PDBs and return their paths
    all_pdbs = sorted(all_pdbs)
    return all_pdbs

def post_process_protein_complex_via_verifiers(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path], max_workers: int = 4) -> List[Path]:
    """
    Return ranked model files for protein complex submission using multiple quality metrics.
    
    This function ranks predictions using a combination of:
    1. iptm score from confidence JSON files
    2. iptm_boltz1 score from confidence JSON files (when available)
    3. OpenMM interaction energy per residue (negated, so lower energy = higher score)
    4. Rosetta interaction energy per residue (negated, so lower energy = higher score)
    5. CDR-Antigen interface fraction (used as a filter for 0 values, not in scoring)
    
    Additionally, any model with a total or heavy-chain CDR interface of 0 is ranked last.
    
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
        max_workers: Maximum number of threads to use for parallel energy calculations
    Returns: 
        Sorted pdb file paths ranked by composite quality score (best first).
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    for prediction_dir in prediction_dirs:
        config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        all_pdbs.extend(config_pdbs)

    if not all_pdbs:
        print("Warning: No PDB files found for ranking")
        return []

    try:
        print(f"Ranking {len(all_pdbs)} PDB files using multi-metric scoring...")
        print(f"Using {max_workers} parallel workers for energy calculations...")
        
        # Calculate scores for each PDB in parallel
        pdb_scores = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pdb = {executor.submit(_calculate_all_metrics, pdb_path): pdb_path for pdb_path in all_pdbs}
            
            # Collect results as they complete
            for future in as_completed(future_to_pdb):
                pdb_path = future_to_pdb[future]
                try:
                    pdb_path_result, composite_score, metric_names = future.result()
                    pdb_scores.append((pdb_path_result, composite_score, metric_names))
                    
                    if metric_names:
                        print(f"  ✓ {pdb_path.name}: score={composite_score:.4f} (metrics: {', '.join(metric_names)})")
                    else:
                        print(f"  ⚠ {pdb_path.name}: score={composite_score:.4f} (no metrics available)")
                        
                except Exception as e:
                    print(f"  ✗ Error processing {pdb_path.name}: {e}")
                    # Add with default score
                    pdb_scores.append((pdb_path, -10000.0, []))
        
        # Sort by composite score (descending - higher score is better)
        pdb_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nRanking results:")
        for i, (pdb_path, score, metrics) in enumerate(pdb_scores):
            print(f"  {i+1}. {pdb_path.name}: {score:.4f} ({', '.join(metrics)})")
        
        # Return sorted PDB paths
        return [pdb_path for pdb_path, _, _ in pdb_scores]

    except Exception as e:
        print(f"\nWARNING: An error occurred during ranking: {e}")
        print("Returning all PDB files unsorted as a fallback.")
        return all_pdbs

def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein-ligand submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    for prediction_dir in prediction_dirs:
        config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        all_pdbs.extend(config_pdbs)
    
    # Sort all PDBs and return their paths
    all_pdbs = sorted(all_pdbs)
    return all_pdbs

# -----------------------------------------------------------------------------
# ---- End of participant section ---------------------------------------------
# -----------------------------------------------------------------------------


DEFAULT_OUT_DIR = Path("predictions")
DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_INPUTS_DIR = Path("inputs")

ap = argparse.ArgumentParser(
    description="Hackathon scaffold for Boltz predictions",
    epilog="Examples:\n"
            "  Single datapoint: python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate\n"
            "  Multiple datapoints: python predict_hackathon.py --input-jsonl examples/test_dataset.jsonl --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-json", type=str,
                        help="Path to JSON datapoint for a single datapoint")
input_group.add_argument("--input-jsonl", type=str,
                        help="Path to JSONL file with multiple datapoint definitions")

ap.add_argument("--msa-dir", type=Path,
                help="Directory containing MSA files (for computing relative paths in YAML)")
ap.add_argument("--submission-dir", type=Path, required=False, default=DEFAULT_SUBMISSION_DIR,
                help="Directory to place final submissions")
ap.add_argument("--intermediate-dir", type=Path, required=False, default=Path("hackathon_intermediate"),
                help="Directory to place generated input YAML files and predictions")
ap.add_argument("--group-id", type=str, required=False, default=None,
                help="Group ID to set for submission directory (sets group rw access if specified)")
ap.add_argument("--result-folder", type=Path, required=False, default=None,
                help="Directory to save evaluation results. If set, will automatically run evaluation after predictions.")
ap.add_argument("--max-workers", type=int, required=False, default=1,
                help="Maximum number of parallel workers for energy calculations in multi-metric ranking (default: 1)")

args = ap.parse_args()

def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligands: Optional[list[SmallMolecule]] = None, msa_dir: Optional[Path] = None) -> dict:
    """
    Prepare input dict for Boltz YAML.
    """
    seqs = []
    for p in proteins:
        if msa_dir and p.msa:
            if Path(p.msa).is_absolute():
                msa_full_path = Path(p.msa)
            else:
                msa_full_path = msa_dir / p.msa
            try:
                msa_relative_path = os.path.relpath(msa_full_path, Path.cwd())
            except ValueError:
                msa_relative_path = str(msa_full_path)
        else:
            msa_relative_path = p.msa
        entry = {
            "protein": {
                "id": p.id,
                "sequence": p.sequence,
                "msa": msa_relative_path
            }
        }
        seqs.append(entry)
    if ligands:
        def _format_ligand(ligand: SmallMolecule) -> dict:
            output =  {
                "ligand": {
                    "id": ligand.id,
                    "smiles": ligand.smiles
                }
            }
            return output
        
        for ligand in ligands:
            seqs.append(_format_ligand(ligand))
    doc = {
        "version": 1,
        "sequences": seqs,
    }
    return doc

def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # Prepare input dict and CLI args
    base_input_dict = _prefill_input_dict(datapoint.datapoint_id, datapoint.proteins, datapoint.ligands, args.msa_dir)

    if datapoint.task_type == "protein_complex":
        configs = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, base_input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        configs = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligands, base_input_dict, args.msa_dir)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    # Run boltz for each configuration
    all_input_dicts = []
    all_cli_args = []
    all_pred_subfolders = []
    
    input_dir = args.intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    for config_idx, (input_dict, cli_args) in enumerate(configs):
        # Write input YAML with config index suffix
        yaml_path = input_dir / f"{datapoint.datapoint_id}_config_{config_idx}.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(input_dict, f, sort_keys=False)

        try:
            # Run boltz
            cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
            fixed = [
                "boltz", "predict", str(yaml_path),
                "--devices", "1",
                "--out_dir", str(out_dir),
                "--cache", cache,
                "--no_kernels",
                "--output_format", "pdb",
            ]
            cmd = fixed + cli_args
            print(f"Running config {config_idx}:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

            # Compute prediction subfolder for this config
            pred_subfolder = out_dir / f"boltz_results_{datapoint.datapoint_id}_config_{config_idx}" / "predictions" / f"{datapoint.datapoint_id}_config_{config_idx}"
            
            all_input_dicts.append(input_dict)
            all_cli_args.append(cli_args)
            all_pred_subfolders.append(pred_subfolder)
        except subprocess.CalledProcessError as e:
            print(f"WARNING: boltz predict failed for config {config_idx} with exit code {e.returncode}. Skipping.")
            print(f"  Command: {' '.join(e.cmd)}")
            continue
        except Exception as e:
            print(f"WARNING: An unexpected error occurred while running config {config_idx}: {e}. Skipping.")
            continue

    # Post-process and copy submissions
    if datapoint.task_type == "protein_complex":
        # Use the new multi-metric ranking function with parallel processing
        ranked_files = post_process_protein_complex_via_verifiers(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders, args.max_workers)
    elif datapoint.task_type == "protein_ligand":
        ranked_files = post_process_protein_ligand(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    if not ranked_files:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    for i, file_path in enumerate(ranked_files[:5]):
        target = subdir / (f"model_{i}.pdb" if file_path.suffix == ".pdb" else f"model_{i}{file_path.suffix}")
        shutil.copy2(file_path, target)
        print(f"Saved: {target}")

    if args.group_id:
        try:
            subprocess.run(["chgrp", "-R", args.group_id, str(subdir)], check=True)
            subprocess.run(["chmod", "-R", "g+rw", str(subdir)], check=True)
        except Exception as e:
            print(f"WARNING: Failed to set group ownership or permissions: {e}")

    # Empty GPU memory and CUDA cache
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

def _load_datapoint(path: Path):
    """Load JSON datapoint file."""
    with open(path) as f:
        return Datapoint.from_json(f.read())

def _run_evaluation(input_file: str, task_type: str, submission_dir: Path, result_folder: Path):
    """
    Run the appropriate evaluation script based on task type.
    
    Args:
        input_file: Path to the input JSON or JSONL file
        task_type: Either "protein_complex" or "protein_ligand"
        submission_dir: Directory containing prediction submissions
        result_folder: Directory to save evaluation results
    """
    script_dir = Path(__file__).parent
    
    if task_type == "protein_complex":
        eval_script = script_dir / "evaluate_abag.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    elif task_type == "protein_ligand":
        eval_script = script_dir / "evaluate_asos.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    print(f"\n{'=' * 80}")
    print(f"Running evaluation for {task_type}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    
    subprocess.run(cmd, check=True)
    print(f"\nEvaluation complete. Results saved to {result_folder}")

def _process_jsonl(jsonl_path: str, msa_dir: Optional[Path] = None):
    """Process multiple datapoints from a JSONL file."""
    print(f"Processing JSONL file: {jsonl_path}")

    for line_num, line in enumerate(Path(jsonl_path).read_text().splitlines(), 1):
        if not line.strip():
            continue

        print(f"\n--- Processing line {line_num} ---")

        try:
            datapoint = Datapoint.from_json(line)
            _run_boltz_and_collect(datapoint)

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON on line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Failed to process datapoint on line {line_num}: {e}")
            raise e
            continue

def _process_json(json_path: str, msa_dir: Optional[Path] = None):
    """Process a single datapoint from a JSON file."""
    print(f"Processing JSON file: {json_path}")

    try:
        datapoint = _load_datapoint(Path(json_path))
        _run_boltz_and_collect(datapoint)
    except Exception as e:
        print(f"ERROR: Failed to process datapoint: {e}")
        raise

def main():
    """Main entry point for the hackathon scaffold."""
    # Determine task type from first datapoint for evaluation
    task_type = None
    input_file = None
    
    if args.input_json:
        input_file = args.input_json
        _process_json(args.input_json, args.msa_dir)
        # Get task type from the single datapoint
        try:
            datapoint = _load_datapoint(Path(args.input_json))
            task_type = datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    elif args.input_jsonl:
        input_file = args.input_jsonl
        _process_jsonl(args.input_jsonl, args.msa_dir)
        # Get task type from first datapoint in JSONL
        try:
            with open(args.input_jsonl) as f:
                first_line = f.readline().strip()
                if first_line:
                    first_datapoint = Datapoint.from_json(first_line)
                    task_type = first_datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    
    # Run evaluation if result folder is specified and task type was determined
    if args.result_folder and task_type and input_file:
        try:
            _run_evaluation(input_file, task_type, args.submission_dir, args.result_folder)
        except Exception as e:
            print(f"WARNING: Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
