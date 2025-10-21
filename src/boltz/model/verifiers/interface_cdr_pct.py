import argparse
import yaml
from pathlib import Path
import os
import sys
from abnumber import Chain
from typing import Dict, List, Set, Tuple

# Add imports for PDB parsing and calculations
import numpy as np
try:
    from Bio.PDB import PDBParser
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O',
}


# I will need a CDR detection library. I'll plan to use anarci.
# from anarci import anarci

def parse_antibody_sequence(sequence):
    """
    Tries to parse a sequence as an antibody chain (single or multi-domain).
    Returns a list of CDR dictionaries, one for each domain found.
    Returns None if the sequence is not a recognizable antibody.
    """
    # This function is designed to suppress the UserWarning from abnumber for ScFvs,
    # as we are explicitly handling both single and multi-domain cases.
    import warnings
    from abnumber.chain import Chain
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            # First, try to parse as a multi-domain sequence, as this is more general.
            # This correctly handles single domains as a list with one item.
            chains = Chain.multiple_domains(sequence, scheme='imgt')
            all_cdrs = []
            for domain in chains:
                cdrs = {
                    "CDR1": domain.cdr1_seq,
                    "CDR2": domain.cdr2_seq,
                    "CDR3": domain.cdr3_seq
                }
                # Only add if CDRs were actually found for the domain
                if any(cdrs.values()):
                    all_cdrs.append({k: v for k, v in cdrs.items() if v is not None})
            
            # Return the list of CDRs if any were found
            if all_cdrs:
                return all_cdrs
        except Exception:
            # If multi-domain parsing fails, it's likely not an antibody.
            # We don't need a separate single-domain check because multiple_domains handles it.
            pass
    return None


class InterfaceCDRPct:
    """
    A verifier to calculate the percentage of antibody CDR residues
    that are at the interface with an antigen.
    """
    def __init__(self, scheme: str = 'imgt', distance_cutoff: float = 8.0):
        """
        Initializes the verifier.
        
        Args:
            scheme: The antibody numbering scheme to use (e.g., 'imgt', 'chothia').
            distance_cutoff: The distance in Angstroms between C-beta atoms to define an interface residue.
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("Biopython is required for InterfaceCDRPct verifier. Please install it using 'pip install biopython'.")
        self.scheme = scheme
        self.distance_cutoff = distance_cutoff

    @staticmethod
    def _get_cb_atom(res):
        """Helper: get CB atom for residue (prefer CB, else CA for GLY)."""
        if res.has_id('CB'):
            return res['CB']
        elif res.get_resname() == 'GLY' and res.has_id('CA'):
            return res['CA']
        else:
            return None

    def get_cdr_residues(self, ab_chain: Chain) -> Set[int]:
        """
        Gets the set of residue numbers corresponding to CDRs from a parsed abnumber Chain.
        This follows the logic from the extract_cdrs.ipynb notebook.
        
        Args:
            ab_chain: An abnumber.Chain object, with ab_chain.pdb_residues attached.
            
        Returns:
            A set of residue numbers (from original PDB numbering).
        """
        cdr_res_nums = set()
        
        # The full sequence from which the ab_chain was derived
        full_seq = ab_chain.seq
        
        # List of PDB residue numbers corresponding to each position in full_seq
        pdb_res_nums = [r.get_id()[1] for r in ab_chain.pdb_residues]

        cdr_sequences = [
            ab_chain.cdr1_seq,
            ab_chain.cdr2_seq,
            ab_chain.cdr3_seq
        ]

        for cdr_seq in cdr_sequences:
            if not cdr_seq:
                continue
            
            start_index = full_seq.find(cdr_seq)
            
            if start_index != -1:
                end_index = start_index + len(cdr_seq)
                # Add all PDB residue numbers within this range
                for i in range(start_index, end_index):
                    if i < len(pdb_res_nums):
                        cdr_res_nums.add(pdb_res_nums[i])

        return cdr_res_nums

    def get_interface_residues(self, structure, antibody_chain_ids: List[str], antigen_chain_ids: List[str]) -> Set[Tuple[str, int]]:
        """
        Identifies antibody residues at the interface with antigen chains based on C-beta distances.
        
        Args:
            structure: A Bio.PDB.Structure object.
            antibody_chain_ids: A list of chain IDs for the antibody.
            antigen_chain_ids: A list of chain IDs for the antigen.
            
        Returns:
            A set of tuples, where each tuple is (chain_id, residue_number) for an interface residue.
        """
        antigen_cb_atoms = []
        for chain_id in antigen_chain_ids:
            if chain_id in structure[0]:
                for res in structure[0][chain_id]:
                    if res.get_id()[0] == ' ': # filter out HETATMs
                        cb = self._get_cb_atom(res)
                        if cb:
                            antigen_cb_atoms.append(cb)
        
        if not antigen_cb_atoms:
            return set()
        
        interface_residues = set()
        for ab_chain_id in antibody_chain_ids:
            if ab_chain_id in structure[0]:
                for ab_residue in structure[0][ab_chain_id]:
                    if ab_residue.get_id()[0] != ' ':
                        continue
                    
                    ab_cb = self._get_cb_atom(ab_residue)
                    if not ab_cb:
                        continue
                        
                    for ag_cb in antigen_cb_atoms:
                        dist = np.linalg.norm(ab_cb.get_coord() - ag_cb.get_coord())
                        if dist < self.distance_cutoff:
                            res_id = ab_residue.get_id()
                            interface_residues.add((ab_chain_id, res_id[1]))
                            break  # Move to the next antibody residue
        
        return interface_residues

    def calculate_cdr_interface_percentage(self, pdb_file_path: str, antibody_chain_ids: List[str]) -> Dict[str, float]:
        """
        Calculates the percentage of CDR residues at the antibody-antigen interface.
        
        Args:
            pdb_file_path: Path to the PDB file of the complex.
            antibody_chain_ids: List of chain IDs corresponding to the antibody (e.g., ['H', 'L']).
            
        Returns:
            A dictionary containing the percentage and counts.
        """
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("complex", pdb_file_path)
            model = structure[0]
        except Exception as e:
            print(f"Error parsing PDB file {pdb_file_path}: {e}")
            return {
                "cdr_interface_pct": 0.0,
                "cdr_residues_count": 0,
                "interface_cdr_residues_count": 0
            }

        all_chain_ids = [chain.id for chain in model]
        antigen_chain_ids = [cid for cid in all_chain_ids if cid not in antibody_chain_ids]

        if not antigen_chain_ids:
            print(f"Warning: No antigen chains found in {pdb_file_path}")
            return {
                "cdr_interface_pct": 0.0,
                "cdr_residues_count": 0,
                "interface_cdr_residues_count": 0
            }

        all_cdr_residues = set()
        for chain_id in antibody_chain_ids:
            if chain_id not in model:
                print(f"    [CDR Finder] Warning: Antibody chain '{chain_id}' not found in PDB model.")
                continue

            chain = model[chain_id]
            
            # Extract sequence and corresponding residue objects from PDB
            pdb_residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ' and r.get_resname() in AA3TO1]
            if not pdb_residues:
                continue
            pdb_seq = "".join([AA3TO1[r.get_resname()] for r in pdb_residues])

            try:
                # Use abnumber to identify CDRs
                ab_chain = Chain(pdb_seq, scheme=self.scheme)
                # Attach PDB residues to the abnumber chain object for later mapping
                ab_chain.pdb_residues = pdb_residues
                
                chain_cdr_residues = self.get_cdr_residues(ab_chain)
                print(f"    [CDR Finder] Found {len(chain_cdr_residues)} CDR residues for chain {chain_id}: {sorted(list(chain_cdr_residues))}")
                for res_num in chain_cdr_residues:
                    all_cdr_residues.add((chain_id, res_num))
            except Exception as e:
                # This may not be an antibody chain, or abnumber failed
                print(f"    [CDR Finder] ERROR: abnumber failed for chain {chain_id}. Sequence length: {len(pdb_seq)}. Error: {e}")
                pass

        interface_residues = self.get_interface_residues(structure, antibody_chain_ids, antigen_chain_ids)
        
        interface_cdr_residues = all_cdr_residues.intersection(interface_residues)
        
        # --- Detailed Per-Chain and Total Calculations ---
        results = {}
        
        # Identify heavy and light chain IDs from the input list
        heavy_chain_id = next((cid for cid in antibody_chain_ids if 'H' in cid.upper()), None)
        light_chain_id = next((cid for cid in antibody_chain_ids if 'L' in cid.upper()), None)

        # Calculate total percentage
        total_cdr_residues_count = len(all_cdr_residues)
        total_interface_cdr_residues_count = len(interface_cdr_residues)
        total_percentage = (total_interface_cdr_residues_count / total_cdr_residues_count * 100) if total_cdr_residues_count > 0 else 0.0
        
        results['cdr_interface_pct'] = total_percentage
        results['cdr_residues_count'] = total_cdr_residues_count
        results['interface_cdr_residues_count'] = total_interface_cdr_residues_count

        # Calculate heavy chain percentage
        if heavy_chain_id:
            heavy_cdr_residues = {res for res in all_cdr_residues if res[0] == heavy_chain_id}
            heavy_interface_cdr_residues = heavy_cdr_residues.intersection(interface_residues)
            heavy_cdr_count = len(heavy_cdr_residues)
            heavy_interface_cdr_count = len(heavy_interface_cdr_residues)
            heavy_percentage = (heavy_interface_cdr_count / heavy_cdr_count * 100) if heavy_cdr_count > 0 else 0.0
            results['heavy_cdr_interface_pct'] = heavy_percentage
            results['heavy_cdr_residues_count'] = heavy_cdr_count
            results['heavy_interface_cdr_residues_count'] = heavy_interface_cdr_count
        
        # Calculate light chain percentage (for completeness)
        if light_chain_id:
            light_cdr_residues = {res for res in all_cdr_residues if res[0] == light_chain_id}
            light_interface_cdr_residues = light_cdr_residues.intersection(interface_residues)
            light_cdr_count = len(light_cdr_residues)
            light_interface_cdr_count = len(light_interface_cdr_residues)
            light_percentage = (light_interface_cdr_count / light_cdr_count * 100) if light_cdr_count > 0 else 0.0
            results['light_cdr_interface_pct'] = light_percentage
            results['light_cdr_residues_count'] = light_cdr_count
            results['light_interface_cdr_residues_count'] = light_interface_cdr_count
        
        return results


def main():
    """
    Main function to extract CDRs from a JSONL file.
    This also serves as an example of how to use the parse_antibody_sequence function.
    """
    parser = argparse.ArgumentParser(description="Extract CDRs from antibody sequences in a JSONL file.")
    parser.add_argument("jsonl_file", type=str, help="Path to a JSONL file containing protein data.")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_file)
    if not jsonl_path.is_file():
        print(f"Error: {jsonl_path} is not a file.")
        return

    import json
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            datapoint_id = data.get('datapoint_id', 'N/A')
            proteins = data.get('proteins', [])
            
            print(f"Datapoint: {datapoint_id}")
            
            found_antibody_chains = {}
            for protein_data in proteins:
                chain_id = protein_data.get('id')
                sequence = protein_data.get('sequence')
                
                # We are interested in Heavy (H) and Light (L) chains for antibodies
                if chain_id in ['H', 'L'] and sequence:
                    cdrs_for_chain = parse_antibody_sequence(sequence)
                    if cdrs_for_chain:
                        found_antibody_chains[chain_id] = cdrs_for_chain

            if not found_antibody_chains:
                print("  No antibody chains (H/L) were identified.")
            else:
                for chain_id, all_cdrs in found_antibody_chains.items():
                    print(f"  Chain {chain_id}:")
                    for i, cdrs in enumerate(all_cdrs):
                        domain_prefix = f"    Domain {i+1}:"
                        if len(all_cdrs) > 1:
                            print(domain_prefix)
                        
                        for cdr_name, cdr_seq in cdrs.items():
                            prefix = "      " if len(all_cdrs) > 1 else "    "
                            print(f"{prefix}{cdr_name}: {cdr_seq}")
            print("-" * 20)


if __name__ == "__main__":
    # Example usage of the InterfaceCDRPct verifier
    # Note: You need a PDB file to run this. 
    # This is a placeholder for how you might use the class.
    # To run the CDR extraction script, provide a JSONL file path as a command-line argument.
    
    if len(sys.argv) > 1:
        main() 
    else:
        print("Running CDR extraction from JSONL file requires a file path.")
        print("Usage: python interface_cdr_pct.py <path_to_jsonl_file>")
        print("\nExample of how to use the InterfaceCDRPct verifier class:")
        
        # Create a dummy PDB file for demonstration if it doesn't exist
        dummy_pdb_path = "dummy_complex.pdb"
        if not os.path.exists(dummy_pdb_path):
            print(f"Creating a dummy PDB file: {dummy_pdb_path}")
            # This is a very minimal PDB format example. A real file would be much more complex.
            dummy_pdb_content = """
ATOM      1  N   ALA A   1      27.340  10.330  -2.470  1.00  0.00           N
ATOM      2  CA  ALA A   1      28.160  10.990  -3.480  1.00  0.00           C
ATOM      3  C   ALA A   1      27.420  12.170  -4.130  1.00  0.00           C
ATOM      4  O   ALA A   1      26.210  12.140  -4.480  1.00  0.00           O
ATOM      5  N   GLY B   1      28.160  13.250  -4.290  1.00  0.00           N
ATOM      6  CA  GLY B   1      27.570  14.450  -4.900  1.00  0.00           C
ATOM      7  C   GLY B   1      28.520  15.220  -5.790  1.00  0.00           C
ATOM      8  O   GLY B   1      29.700  14.920  -5.910  1.00  0.00           O
"""
            with open(dummy_pdb_path, 'w') as f:
                f.write(dummy_pdb_content.strip())

        try:
            verifier = InterfaceCDRPct()
            # In a real scenario, you would have ['H', 'L'] as antibody chains
            antibody_chains = ['A'] 
            result = verifier.calculate_cdr_interface_percentage(dummy_pdb_path, antibody_chains)
            print(f"Verifier results for dummy PDB: {result}")
        except ImportError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred during verifier test: {e}")

"""
I have to fix the main function. Right now it simply extract the cdr only from heavy chain. We have to fix it to extract cdrs for both heavy and light chain. Also, it has to use hackathon_data/datasets/abag_public/abag_public.jsonl to get sequences, not yaml files
""" 