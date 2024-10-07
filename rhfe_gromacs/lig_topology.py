import os
import re

import parmed as pmd
from openff.toolkit import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import app


def unique_atom_types(pmd_structure, name):
    omm_system = pmd_structure.createSystem(nonbondedMethod=app.NoCutoff,
                                                constraints=None,
                                                removeCMMotion=False,
                                                rigidWater=False)
    topology = pmd_structure.topology
    positions = pmd_structure.positions
    count_id = 1
    for c in topology.chains():
        for r in c.residues():
            r.name = name
            for a in r.atoms():
                if r.name == name:
                    a.id = ligand.name + a.name[0] + str(count_id)
                    a.id = a.id[:6]
                count_id += 1
    new_system_structure = pmd.openmm.load_topology(topology,
                                                        system=omm_system,
                                                        xyz=positions)
    new_system_structure.positions = pmd_structure.positions
    new_system_structure.velocities = pmd_structure.velocities
    new_system_structure.box = pmd_structure.box

    return new_system_structure


if __name__ == "__main__":
    sets = range(1,8)

    for lig_set in sets:
        ligands = Molecule.from_file(f'input/ligands/aligned_{lig_set}.sdf')

        # Load a forcefield
        lig_ff = ForceField('openff_unconstrained-2.1.1.offxml')

        for ligand in ligands:
            ligand.name = ligand.name[:3]
            ligand_topology = ligand.to_topology()

            ligand_positions = ligand.conformers[0]
            ligand_topology = ligand.to_topology()
            ligand_system = lig_ff.create_openmm_system(ligand_topology)

            pmd_ligand_struct = pmd.openmm.load_topology(
                ligand_topology.to_openmm(), ligand_system, ligand_positions)

            new_ligand_struct = unique_atom_types(pmd_ligand_struct, ligand.name)

            if not os.path.exists(f'input/ligands/lig_{ligand.name}'):
                os.mkdir(f'input/ligands/lig_{ligand.name}')

            new_ligand_struct.save(f'input/ligands/lig_{ligand.name}/mol.top', overwrite=True)
            new_ligand_struct.save(f'input/ligands/lig_{ligand.name}/mol_gmx.pdb',
                                overwrite=True)

            with open(f'input/ligands/lig_{ligand.name}/mol_gmx.pdb', 'r+') as f:
                content = f.read()
                content_new = re.sub('([A-Z]{1})(\d{1}|\d{2})(x)',
                                    r'\1\2 ',
                                    content,
                                    flags=re.M)
                # content_new = re.sub('HETATM', 'ATOM  ', content_new, flags = re.M)
                f.seek(0)
                f.write(content_new)

            with open(f'input/ligands/lig_{ligand.name}/mol.top', 'r+') as f:
                content = f.read()
                atomtypes = re.search(r'(\[ atomtypes \])((.|\n)*)',
                                    content).group().split('[ moleculetype ]')[0]
                content_new = content.replace(f"{ligand.name}          3", "MOL          3")
                content_new = content_new.replace('1               2               no              1            0.83333333  ', '1               2               yes             0.5          0.83333333  ')
                f.seek(0)
                f.write(content_new)

            with open(f'input/ligands/lig_{ligand.name}/ffMOL.itp', 'w') as f:
                f.write(atomtypes)
