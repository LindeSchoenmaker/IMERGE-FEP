# setting chirality based on https://www.valencekjell.com/posts/2021-10-15-chiral-templating/index.html

import itertools
import warnings
from collections import defaultdict
from itertools import product

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdRGroupDecomposition as rdRGD
from rdkit.Chem.rdchem import EditableMol

from rgroupinterm.utils.sanifix import AdjustAromaticNs

rdBase.DisableLog('rdApp.*')


class EnumRGroups():
    """Class for R-group enumeration.

    This class enumerates all combinations of R-groups for a molecule pair.

    Attributes:
        multiple (bool): if multiple r-groups have been found for the molecule pair
        columns (str): column names of columns containing r-groups
    """

    def __init__(self):
        """The init."""
        self.multiple = False
        self.columns = ['R1']

    def generate_intermediates(self,
                               pair: list[Chem.rdchem.Mol]) -> pd.DataFrame:
        """Method that combines functions in this class to generate intermediates with enumerated r-groups.

        Args:
            pair list[Chem.rdchem.Mol]: pair of structures to generate intermediates for, list of two rdkit molecule objects

        Returns:
            df_interm (pd.DataFrame): dataframe with the generated intermedaites
        """
        self.pair = pair
        self.df = self.get_rgroups()
        self.df_interm = self.enumerate_rgroups()
        self.weld()
        self.clean_intermediates()

        #TODO clean up what to return, can also just be a list
        return self.df_interm, self.df_rgroup['Core'][0]

    def get_rgroups(self) -> pd.DataFrame:
        """Method that determines common core and r-groups.

        Use maximum common substructure of two molecules to get the differing r-group(s).
        Sets flag "multiple" to true if there are multiple r-groups.
        Ensures that fused ring r-groups are not present multiple times.

        Returns:
            df_rgroup (pd.DataFrame): dataframe with molecule objects for Core, R1, ... Rn
        """
        # make aromatic bonds explicit
        for i, mol in enumerate(self.pair):
            Chem.Kekulize(mol, clearAromaticFlags=True)
            for atom in mol.GetAtoms():
                atom.SetIntProp("SourceAtomIdx", atom.GetIdx())
                atom.SetIntProp("SourceMol", i)

        #TODO possible to use different comparison functions
        # find maximimum common substructure
        res = rdFMCS.FindMCS(self.pair,
                             matchValences=True,
                             ringMatchesRingOnly=True,
                             completeRingsOnly=True,
                             timeout=2)
        core = Chem.MolFromSmarts(res.smartsString)

        new_pair = self._set_idx_sameforcore(core)

        self.pair = new_pair
        # create dataframe with columns Core, R1, ... Rn
        res, _ = rdRGD.RGroupDecompose([core],
                                       self.pair,
                                       asSmiles=False,
                                       asRows=False)
        self.df_rgroup = pd.DataFrame(res)

        # in case of multiple core matches, pick atom with right core idxs after rgroupdecomposition
        todrop = []
        for index, row in self.df_rgroup.iterrows():
            mol = row['Core']
            for atom in mol.GetAtoms():
                if atom.HasProp('SourceAtomIdxSameCore'):
                    if atom.GetIntProp(
                            'SourceAtomIdxSameCore') > self.core_length:
                        todrop.append(index)

        self.df_rgroup = self.df_rgroup.drop(todrop).reset_index(drop=True)
        self.pair = [ele for i, ele in enumerate(self.pair) if i not in todrop]

        self.stereocenters = defaultdict(dict)
        # dict with at each r-group branching off point the order of the bonds (used for setting chirality)
        self.bondorder = defaultdict(dict)
        # need to do twice because core has different atom idx based on input
        #TODO may not be necessary anymore to do this twice as also have version with same Idx now
        for i, mol in enumerate(self.pair):
            res_core = self.df_rgroup['Core'][i]
            res_core_copy = Chem.Mol(
                res_core
            )  # otherwise problem with attachment point connected to multiple groups
            mapping_dict = self._get_source_mapping(res_core_copy)
            for key, value in mapping_dict.items():
                self.stereocenters[i][key] = mol.GetAtomWithIdx(
                    value).GetChiralTag()
                bond_order = []
                for bond in mol.GetAtomWithIdx(value).GetBonds():
                    if bond.GetBeginAtomIdx(
                    ) == value:  #only keep order of source id of atoms connected (not the pair as done in example)
                        bond_order.append(bond.GetEndAtom().GetIntProp(
                            "SourceAtomIdxSameCore"))
                    else:
                        bond_order.append(bond.GetBeginAtom().GetIntProp(
                            "SourceAtomIdxSameCore"))
                self.bondorder[i][key] = bond_order

        #TODO check if does anything
        # adjust aromatic Ns to [nH] if needed if core is not a valid SMILES
        res_core = self.df_rgroup['Core'][0]
        if Chem.MolFromSmiles(Chem.MolToSmiles(res_core)) is None:
            adj_core = AdjustAromaticNs(res_core)
            if adj_core:
                self.df_rgroup['Core'][0] = adj_core

        # set multiple to true if more than 1 r-group
        if len(self.df_rgroup.columns) > 2:
            self.multiple = True
            self.columns = self.df_rgroup.columns[
                1:]  #TODO change to take everything that starts with R # update column names

        # # remove duplicate r-groups
        if self.multiple:
            same_dict = defaultdict(
                set,
                {k: set()
                 for k in (0, 1)})  # per row, column name of duplicate r-group
            combine_columns = set(
            )  # names of columns in which a duplicate r-group is present
            column_combs = list(
                itertools.combinations(self.columns.tolist(), 2))
            for index, row in self.df_rgroup.iterrows():
                for column_comb in column_combs:
                    # check is smiles are identical
                    if Chem.MolToSmiles(
                            row[column_comb[0]]) == Chem.MolToSmiles(
                                row[column_comb[1]]):
                        same_dict[index].add(column_comb[1])
                        combine_columns.update(column_comb)
            if len(combine_columns) > 0:
                for idx, same in same_dict.items():
                    to_combine_set = combine_columns.difference(
                        same)  # column names with unique r-groups to combine
                    to_combine = list(to_combine_set)
                    all_columns = list(combine_columns)
                    # iterate over r-groups to combine to create new molecule object that contains all
                    mols_combined = self.df_rgroup.at[idx, to_combine[0]]
                    for column in to_combine[1:]:
                        mols_combined = Chem.CombineMols(
                            mols_combined, self.df_rgroup.at[idx, column])
                    self.df_rgroup.at[idx, all_columns[0]] = mols_combined
                # drop columns containing the duplicate entries
                self.df_rgroup = self.df_rgroup.drop(
                    columns=list(combine_columns)[1:])
                self.columns = self.df_rgroup.columns[
                    1:]  # update r-group column names

        return self.df_rgroup

    def _set_idx_sameforcore(self, core):
        # get indices of the molecule’s atoms that match the core
        matches = []
        for mol in self.pair:
            match = mol.GetSubstructMatches(core,
                                            useQueryQueryMatches=False,
                                            useChirality=True)
            if len(match) == 1:
                matches.append(list(match[0]))
            elif len(match) > 1:
                # there are multiple possible ways the core can be matched to the parent molecule
                matches.append(match)
            else:
                warnings.warn(
                    'No matches between core and one of the parent molecules')

        # set SourceAtomIdx of atoms in the core to the same number
        new_pair = []
        highest_idx = None
        for match in matches:
            if isinstance(match, list):
                self.core_length = len(match)
                highest_idx = len(match)
                break
        if highest_idx is None:
            print("add way to add highest idx if both return multiple matches")
        for i, mol in enumerate(self.pair):
            if isinstance(matches[i], list):
                new_mol = Chem.Mol(mol)
                for j, k in enumerate(matches[i]):
                    new_mol.GetAtomWithIdx(k).SetProp('SourceAtomIdxSameCore',
                                                      str(j))
                for atom in new_mol.GetAtoms():
                    if not atom.HasProp('SourceAtomIdxSameCore'):
                        highest_idx += 1
                        atom.SetProp('SourceAtomIdxSameCore', str(highest_idx))
                new_pair.append(new_mol)
            else:
                for match in matches[i]:
                    new_mol = Chem.Mol(mol)
                    for j, k in enumerate(list(match)):
                        new_mol.GetAtomWithIdx(k).SetProp(
                            'SourceAtomIdxSameCore', str(j))
                    for atom in new_mol.GetAtoms():
                        if not atom.HasProp('SourceAtomIdxSameCore'):
                            highest_idx += 1
                            atom.SetProp('SourceAtomIdxSameCore',
                                         str(highest_idx))
                    new_pair.append(new_mol)
        return new_pair

    def _get_source_mapping(self, input_mol):
        join_dict = {}
        for atom in input_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                join_dict[map_num] = (atom)

        # transfer the atom maps to the neighbor atoms
        for idx, atom in join_dict.items():
            nbr_1 = [x.GetOtherAtom(atom) for x in atom.GetBonds()][0]
            nbr_1.SetAtomMapNum(idx)

        # remove the dummy atoms
        new_mol = Chem.DeleteSubstructs(input_mol, Chem.MolFromSmarts('[#0]'))

        # get the new atoms with AtomMapNum, these will be connected
        source_atom_dict = {}
        for atom in new_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                if atom.HasProp("SourceAtomIdx"):
                    source_atom_dict[map_num] = atom.GetIntProp(
                        "SourceAtomIdx")

        return source_atom_dict

    def enumerate_rgroups(self) -> pd.DataFrame:
        """Enumerate all possible r-group combinations

        Returns:
            df_interm (pd.DataFrame): dataframe with molecule objects for Core, R1, ... Rn of enumerated r-groups
        """
        # combinations of r-groups at the same positions
        combinations = list(
            product(*tuple([self.df[column]
                            for column in self.df.columns[1:]])))
        # create new dataframe for storing adapted rgroups [has Core, R1, ... Rn]
        self.df_interm = pd.DataFrame(combinations,
                                      columns=self.df.columns[1:])
        self.df_interm['Core'] = self.df.at[0, 'Core']

        return self.df_interm

    def weld(self):
        """Put modified rgroups back on the core.
        
        Returns:
            df_interm (pd.DataFrame): now with column Intermediate, contains SMILES of the put together intermediate
        """
        #TODO maybe just return those as a list
        self.df_interm['Intermediate'] = None
        for index, row in self.df_interm.iterrows():
            #TODO check if try does anything
            try:
                mol_to_weld = row['Core']
                for column in self.columns:
                    mol_to_weld = Chem.CombineMols(mol_to_weld, row[column])
                welded_mol = self._weld_r_groups(mol_to_weld)
                if welded_mol:
                    self.df_interm.at[
                        index, 'Intermediate'] = Chem.MolToSmiles(welded_mol)
                else:
                    self.df_interm.at[index, 'Intermediate'] = None
            # except AttributeError:
            #     pass
            except IndexError:
                pass
            except Chem.rdchem.AtomKekulizeException:
                pass

    def _weld_r_groups(self, input_mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
        """RDKit manipulations to put r-groups back on a core.

        Adapted from 
        https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/CANPfuvAqWxR%2BosdH6TUT-%2B1Fy85fUXh0poRddrEQDxXmguJJ7Q%40mail.gmail.com/
        AtomMapNum: number to keep track of what is connected to what
        Args:
            input_mol (Chem.rdchem.Mol): combined mol object of core and r-groups

        Returns:
            final_mol (Chem.rdchem.Mol|None): welded molecule
        """
        # loop over atoms and find the atoms with an AtomMapNum
        join_dict = defaultdict(
            list)  # list of atoms to connect for each r-group
        for atom in input_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                join_dict[map_num].append(atom)

        # loop over the bond between atom and dummy atom of r-group and save bond type
        bond_order_dict = defaultdict(list)
        chiral_tag_dict = {
        }  # dict to store chiral tag of the source of the branching off points
        source_bond_order_dict = {
        }  # dict to store bond order in the source of the branching off points
        for idx, atom_list in join_dict.items():
            if len(atom_list) == 2:
                atm_1, atm_2 = atom_list
                bond_order_dict[idx].append(atm_2.GetBonds()[0].GetBondType())
            elif len(atom_list) == 3:
                atm_1, atm_2, atm_3 = atom_list
                bond_order_dict[idx].append(atm_2.GetBonds()[0].GetBondType())
                bond_order_dict[idx].append(atm_3.GetBonds()[0].GetBondType())
            elif len(atom_list) == 4:
                atm_1, atm_2, atm_3, atm_4 = atom_list
                bond_order_dict[idx].append(atm_2.GetBonds()[0].GetBondType())
                bond_order_dict[idx].append(atm_3.GetBonds()[0].GetBondType())
                bond_order_dict[idx].append(atm_4.GetBonds()[0].GetBondType())
        # transfer the atom maps to the neighbor atoms
        for idx, atom_list in join_dict.items():
            if len(atom_list) == 2:
                atm_1, atm_2 = atom_list
                nbr_1 = [x.GetOtherAtom(atm_1) for x in atm_1.GetBonds()][0]
                nbr_1.SetAtomMapNum(idx)
                nbr_2 = [x.GetOtherAtom(atm_2) for x in atm_2.GetBonds()][0]
                nbr_2.SetAtomMapNum(idx)
            elif len(atom_list) == 3:
                atm_1, atm_2, atm_3 = atom_list
                nbr_1 = [x.GetOtherAtom(atm_1) for x in atm_1.GetBonds()][0]
                nbr_1.SetAtomMapNum(idx)
                nbr_2 = [x.GetOtherAtom(atm_2) for x in atm_2.GetBonds()][0]
                nbr_2.SetAtomMapNum(idx)
                nbr_3 = [x.GetOtherAtom(atm_3) for x in atm_3.GetBonds()][0]
                nbr_3.SetAtomMapNum(idx)
            elif len(atom_list) == 4:
                atm_1, atm_2, atm_3, atm_4 = atom_list
                nbr_1 = [x.GetOtherAtom(atm_1) for x in atm_1.GetBonds()][0]
                nbr_1.SetAtomMapNum(idx)
                nbr_2 = [x.GetOtherAtom(atm_2) for x in atm_2.GetBonds()][0]
                nbr_2.SetAtomMapNum(idx)
                nbr_3 = [x.GetOtherAtom(atm_3) for x in atm_3.GetBonds()][0]
                nbr_3.SetAtomMapNum(idx)
                nbr_4 = [x.GetOtherAtom(atm_4) for x in atm_4.GetBonds()][0]
                nbr_4.SetAtomMapNum(idx)
            if nbr_2.HasProp("SourceMol"):
                chiral_tag_dict[idx] = self.stereocenters[nbr_2.GetIntProp(
                    "SourceMol")][idx]
                source_bond_order_dict[idx] = self.bondorder[nbr_2.GetIntProp(
                    "SourceMol")][idx]
            else:
                chiral_tag_dict[idx] = Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                source_bond_order_dict[idx] = None
        # remove the dummy atoms
        new_mol = Chem.DeleteSubstructs(input_mol, Chem.MolFromSmarts('[#0]'))

        # get the new atoms with AtomMapNum, these will be connected
        bond_join_dict = defaultdict(list)
        for atom in new_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                bond_join_dict[map_num].append(atom.GetIdx())

        # make an editable molecule and add bonds between atoms with correspoing AtomMapNum
        em = EditableMol(new_mol)
        for idx, atom_list in bond_join_dict.items():
            if len(atom_list) == 2:
                start_atm, end_atm = atom_list
                em.AddBond(start_atm, end_atm, order=bond_order_dict[idx][0])
            elif len(atom_list) == 3:
                start_atm, end_atm1, end_atm2 = atom_list
                em.AddBond(start_atm, end_atm1, order=bond_order_dict[idx][0])
                em.AddBond(start_atm, end_atm2, order=bond_order_dict[idx][1])
            elif len(atom_list) == 4:
                start_atm, end_atm1, end_atm2, end_atm3 = atom_list
                em.AddBond(start_atm, end_atm1, order=bond_order_dict[idx][0])
                em.AddBond(start_atm, end_atm2, order=bond_order_dict[idx][1])
                em.AddBond(start_atm, end_atm3, order=bond_order_dict[idx][2])

        mol_new = em.GetMol()

        # make edible molecule to set bond orders (used for correctly applying chiral tag)
        rw_mol = Chem.RWMol(mol_new)
        for idx, atom_list in bond_join_dict.items():
            if chiral_tag_dict[idx] == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                continue
            bond_info = []
            bond_order_curr = []
            for bond in rw_mol.GetAtomWithIdx(atom_list[0]).GetBonds():
                if bond.GetBeginAtomIdx() == atom_list[
                        0]:  #only keep order of source id of atoms connected
                    new_id = bond.GetEndAtomIdx()
                else:
                    new_id = bond.GetBeginAtomIdx()
                if rw_mol.GetAtomWithIdx(new_id).HasProp(
                        "SourceAtomIdxSameCore"):
                    source_id = rw_mol.GetAtomWithIdx(new_id).GetIntProp(
                        "SourceAtomIdxSameCore")  # use original source idx
                    if source_id not in bond_order_curr:
                        bond_order_curr.append(
                            source_id
                        )  # save current bond order to be used to retrieve information
                bond_info.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                                  bond.GetBondType()))
                rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            # set new bond order
            order = [
                source_bond_order_dict[idx].index(source_id)
                for source_id in bond_order_curr
            ]
            for i in order:
                rw_mol.AddBond(*bond_info[i])

        combined_mol = rw_mol.GetMol()

        # set chiral tags
        for idx, atom_list in bond_join_dict.items():
            combined_mol.GetAtomWithIdx(atom_list[0]).SetChiralTag(
                chiral_tag_dict[idx])

        # postprocessing to fix number of hydrogens at attachment points
        for atom in combined_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num == 0: continue
            if atom.GetAtomicNum(
            ) == 7 and not atom.IsInRing():  # for nitrogen atoms
                atom.SetNoImplicit(True)
                nbrs = list(atom.GetNeighbors())
                nonHs = [nbr.GetAtomicNum() != 1 for nbr in nbrs]
                bonds = list(atom.GetBonds())
                bondtypes = [bond.GetBondType() for bond in bonds]
                i = 0
                for bondtype in bondtypes:
                    if bondtype == Chem.BondType.DOUBLE:
                        i += 1
                    elif bondtype == Chem.BondType.TRIPLE:
                        i += 2
                atom.SetNumExplicitHs(3 - len(nonHs) - i +
                                      atom.GetFormalCharge())
            if atom.GetAtomicNum(
            ) == 6 and not atom.IsInRing():  # for carbon atoms
                atom.SetNoImplicit(True)
                nbrs = list(atom.GetNeighbors())
                nonHs = [nbr.GetAtomicNum() != 1 for nbr in nbrs]
                bonds = list(atom.GetBonds())
                bondtypes = [bond.GetBondType() for bond in bonds]
                i = 0
                for bondtype in bondtypes:
                    if bondtype == Chem.BondType.DOUBLE:
                        i += 1
                    elif bondtype == Chem.BondType.TRIPLE:
                        i += 2
                atom.SetNumExplicitHs(4 - len(nonHs) - i +
                                      atom.GetFormalCharge())
        combined_mol_fixed = combined_mol

        # if molecule is invalid try replacing single bond tokens
        if Chem.MolFromSmiles(Chem.MolToSmiles(combined_mol)) is None:
            combined_mol_fixed = Chem.MolFromSmiles(
                Chem.MolToSmiles(combined_mol).replace('-', ''))
            if combined_mol_fixed is None:
                print(f'invalid molecule: {Chem.MolToSmiles(combined_mol)}')
            else:
                combined_mol = combined_mol_fixed

        # remove the AtomMapNum values
        for atom in combined_mol.GetAtoms():
            atom.SetAtomMapNum(0)

        # remove explicit Hs
        final_mol = Chem.RemoveHs(combined_mol)

        #TODO maybe do somewhere else
        # restore bonds to aromatic type
        Chem.SanitizeMol(final_mol)

        return final_mol

    def clean_intermediates(self):
        """ 
        Drop duplicates & intermediates that are same as molecules in pair.
        Drops columns that are not needed anymore

        Get intermediate SMILES
        """
        # drop NoneType SMILES
        self.df_interm = self.df_interm.dropna(subset=['Intermediate'])

        if len(self.df_interm) > 0:
            #remove intermediates that are same as original pair
            for mol in self.pair:
                Chem.SanitizeMol(mol)
            parent_smiles = list(map(Chem.MolToSmiles, self.pair))
            self.df_interm['Exists'] = self.df_interm.apply(
                lambda row: str(parent_smiles.index(row.Intermediate)) if row.Intermediate in parent_smiles else False,
                axis=1)
            # assert if parent molecules are recreated (if not, indicative of unexpected behaviour)
            assert '0' in self.df_interm['Exists'].values.tolist()
            assert '1' in self.df_interm['Exists'].values.tolist()
            self.df_interm = self.df_interm[self.df_interm['Exists'] == False]
            self.df_interm = self.df_interm.drop(columns=['Exists'])

        if len(self.df_interm) > 0:
            # return mol objects
            self.df_interm['Intermediate'] = self.df_interm.apply(
                lambda row: Chem.MolFromSmiles(row.Intermediate), axis=1)

        self.df_interm = self.df_interm.drop(
            columns=[*['Core'], *self.columns])
        self.df_interm = self.df_interm.dropna()
