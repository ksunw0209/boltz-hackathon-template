import pyrosetta
from pyrosetta import pose_from_pdb, get_score_function

class RosettaInteractionEnergy:
    def __init__(self, scorefxn_name='ref2015'):
        """
        Initializes the RosettaInteractionEnergy calculator.

        Parameters
        ----------
        scorefxn_name : str
            Name of the Rosetta score function to use.
        """
        # In older pyrosetta versions, is_initialized() does not exist.
        # It is safe and necessary to call init() in each worker process.
        pyrosetta.init(extra_options="-ignore_unrecognized_res true -ex1 -ex2 -mute all")
        self.scorefxn = get_score_function(True) # Using the default full-atom score function, typically ref2015

    def calculate_interaction_energy(self, pdb_file_path, return_force=False):
        """
        Calculates the interaction energy between chains in a PDB file using Rosetta.

        This method is designed to be parallelizable. You can call this method
        in parallel for different PDB files.

        Parameters
        ----------
        pdb_file_path : str
            Path to the PDB file.
        return_force : bool, optional
            If True, return the forces as well. Not yet implemented.

        Returns
        -------
        dict
            A dictionary containing:
            - 'interaction_energy': The total interaction energy in Rosetta Energy Units (REU).
            - 'energy_per_residue': The interaction energy normalized by the total number of residues.
        """
        if return_force:
            raise NotImplementedError("Force calculation is not yet implemented for Rosetta.")

        try:
            pose = pose_from_pdb(str(pdb_file_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load PDB file {pdb_file_path}: {e}")

        complex_energy = self.scorefxn(pose)

        total_chain_energy = 0.0
        
        chains = pose.split_by_chain()
        
        if len(chains) <= 1:
            interaction_energy = 0.0
        else:
            for chain_pose in chains:
                if chain_pose.total_residue() > 0:
                    chain_energy = self.scorefxn(chain_pose)
                    total_chain_energy += chain_energy
            interaction_energy = complex_energy - total_chain_energy

        total_residues = pose.total_residue()
        
        complex_energy_per_residue = complex_energy / total_residues if total_residues > 0 else 0.0
        interaction_energy_per_residue = interaction_energy / total_residues if total_residues > 0 else 0.0

        return {
            'complex_energy': complex_energy,
            'interaction_energy': interaction_energy,
            'complex_energy_per_residue': complex_energy_per_residue,
            'interaction_energy_per_residue': interaction_energy_per_residue
        }

    def calculate_complex_energy(self, pdb_file_path, return_force=False):
        """
        Calculates the complex energy of a PDB file using Rosetta.

        Parameters
        ----------
        pdb_file_path : str
            Path to the PDB file.
        return_force : bool, optional
            If True, return the forces as well. Not yet implemented.

        Returns
        -------
        dict
            A dictionary containing:
            - 'complex_energy': The total energy of the complex in Rosetta Energy Units (REU).
            - 'energy_per_residue': The complex energy normalized by the total number of residues.
        """
        if return_force:
            raise NotImplementedError("Force calculation is not yet implemented for Rosetta.")

        try:
            pose = pose_from_pdb(str(pdb_file_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load PDB file {pdb_file_path}: {e}")

        complex_energy = self.scorefxn(pose)

        total_residues = pose.total_residue()
        energy_per_residue = complex_energy / total_residues if total_residues > 0 else 0.0

        return {
            'complex_energy': complex_energy,
            'energy_per_residue': energy_per_residue
        }

