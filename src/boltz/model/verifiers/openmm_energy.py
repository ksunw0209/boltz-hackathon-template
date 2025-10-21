import io
from openmm.app import ForceField, PDBFile, Modeller, NoCutoff, CutoffNonPeriodic, HBonds
from pdbfixer import PDBFixer
from openmm import LangevinIntegrator, Context, Platform
from openmm.unit import nanometers, kelvin, picosecond, kilojoules_per_mole, picoseconds
import time
import numpy as np

class OpenMMInteractionEnergy:
    def __init__(self, forcefield_files=['amber14-all.xml', 'tip3pfb.xml']):
        """
        Initializes the OpenMMInteractionEnergy calculator.

        Parameters
        ----------
        forcefield_files : list of str
            List of forcefield XML files to use.
        """
        self.forcefield = ForceField(*forcefield_files)

    def _get_energy(self, topology, positions, platform, properties=None, get_forces=False):
        """
        Helper function to calculate the potential energy of a system.
        """
        system = self.forcefield.createSystem(topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        integrator.setRandomNumberSeed(0)  # For deterministic results
        if properties:
            context = Context(system, integrator, platform, properties)
        else:
            context = Context(system, integrator, platform)
        context.setPositions(positions)
        state = context.getState(getEnergy=True, getForces=get_forces)
        energy = state.getPotentialEnergy()
        forces = None
        if get_forces:
            forces = state.getForces(asNumpy=True)
        return energy, forces

    def calculate_interaction_energy(self, pdb_file_path, platform_name='CUDA', device_index=None, return_force=False, use_fixer=True):
        """
        Calculates the interaction energy between chains in a PDB file.

        This method is designed to be parallelizable. You can call this method
        in parallel for different PDB files.

        Parameters
        ----------
        pdb_file_path : str
            Path to the PDB file.
        platform_name : str
            Name of the OpenMM platform to use ('CPU', 'CUDA', 'OpenCL').
        device_index : int, optional
            Index of the GPU device to use for the calculation.
        return_force : bool, optional
            If True, return the forces as well.
        use_fixer : bool, optional
            If True, use PDBFixer to repair the PDB file before calculation.

        Returns
        -------
        dict
            A dictionary containing:
            - 'interaction_energy': The total interaction energy in kJ/mol.
            - 'energy_per_residue': The interaction energy normalized by the total number of residues.
            - 'forces': The interaction forces if return_force is True.
        """
        platform = Platform.getPlatformByName(platform_name)
        properties = {}
        if device_index is not None and platform_name in ['CUDA', 'OpenCL']:
            properties['DeviceIndex'] = str(device_index)

        if use_fixer:
            # Use PDBFixer to repair the PDB file. This is more robust than Modeller
            # for fixing issues commonly found in structures from prediction tools.
            fixer = PDBFixer(filename=pdb_file_path)
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(False)  # This removes water and other ligands
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            modeller = Modeller(fixer.topology, fixer.positions)
        else:
            pdb = PDBFile(pdb_file_path)
            modeller = Modeller(pdb.topology, pdb.positions)

        # Energy of the whole complex using the corrected model
        complex_topology = modeller.topology
        complex_positions = modeller.positions
        complex_energy_start_time = time.time()
        complex_energy, complex_forces = self._get_energy(complex_topology, complex_positions, platform, properties, get_forces=return_force)
        print(f"Complex energy calculation time: {time.time() - complex_energy_start_time} seconds")

        # Sum of energies of individual chains
        total_chain_energy = 0.0 * kilojoules_per_mole
        interaction_forces = complex_forces.copy() if return_force and complex_forces is not None else None
        
        chains = list(complex_topology.chains())
        total_chain_energy_start_time = time.time()
        for i in range(len(chains)):
            # Create a new Modeller for the single chain from the hydrogen-added complex
            chain_modeller = Modeller(complex_topology, complex_positions)
            
            # Keep only the i-th chain
            chains_to_delete = [c for j, c in enumerate(chain_modeller.topology.chains()) if i != j]
            chain_modeller.delete(chains_to_delete)
            
            if chain_modeller.topology.getNumAtoms() > 0:
                chain_energy, chain_forces = self._get_energy(chain_modeller.topology, chain_modeller.positions, platform, properties, get_forces=return_force)
                total_chain_energy += chain_energy
                if return_force and interaction_forces is not None and chain_forces is not None:
                    chain_atom_indices = [atom.index for atom in chains[i].atoms()]
                    for k, original_atom_index in enumerate(chain_atom_indices):
                        interaction_forces[original_atom_index] -= chain_forces[k]

        print(f"Total chain energy calculation time: {time.time() - total_chain_energy_start_time} seconds")

        interaction_energy = complex_energy - total_chain_energy
        
        total_residues = modeller.topology.getNumResidues()
        
        complex_energy_per_residue = complex_energy / total_residues if total_residues > 0 else 0.0
        interaction_energy_per_residue = interaction_energy / total_residues if total_residues > 0 else 0.0

        output = {
            'complex_energy': complex_energy,
            'interaction_energy': interaction_energy,
            'complex_energy_per_residue': complex_energy_per_residue,
            'interaction_energy_per_residue': interaction_energy_per_residue
        }
        if return_force and interaction_forces is not None:
            output['forces'] = interaction_forces.value_in_unit(kilojoules_per_mole/nanometers).tolist()
        return output

    def calculate_complex_energy(self, pdb_file_path, platform_name='CUDA', device_index=None, return_force=False, use_fixer=True):
        """
        Calculates the complex energy of a PDB file using OpenMM.

        Parameters
        ----------
        pdb_file_path : str
            Path to the PDB file.
        platform_name : str
            Name of the OpenMM platform to use ('CPU', 'CUDA', 'OpenCL').
        device_index : int, optional
            Index of the GPU device to use for the calculation.
        return_force : bool, optional
            If True, return the forces as well.
        use_fixer : bool, optional
            If True, use PDBFixer to repair the PDB file before calculation.

        Returns
        -------
        dict
            A dictionary containing:
            - 'complex_energy': The total energy of the complex in kJ/mol.
            - 'energy_per_residue': The complex energy normalized by the total number of residues.
            - 'forces': The forces if return_force is True.
        """
        platform = Platform.getPlatformByName(platform_name)
        properties = {}
        if device_index is not None and platform_name in ['CUDA', 'OpenCL']:
            properties['DeviceIndex'] = str(device_index)

        if use_fixer:
            fixer = PDBFixer(filename=pdb_file_path)
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(False)
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            modeller = Modeller(fixer.topology, fixer.positions)
        else:
            pdb = PDBFile(pdb_file_path)
            modeller = Modeller(pdb.topology, pdb.positions)

        complex_topology = modeller.topology
        complex_positions = modeller.positions
        complex_energy, complex_forces = self._get_energy(complex_topology, complex_positions, platform, properties, get_forces=return_force)
        
        total_residues = modeller.topology.getNumResidues()
        energy_per_residue = complex_energy / total_residues if total_residues > 0 else 0.0 * kilojoules_per_mole

        output = {
            'complex_energy': complex_energy,
            'energy_per_residue': energy_per_residue
        }
        if return_force and complex_forces is not None:
            output['forces'] = complex_forces.value_in_unit(kilojoules_per_mole/nanometers).tolist()
        return output
