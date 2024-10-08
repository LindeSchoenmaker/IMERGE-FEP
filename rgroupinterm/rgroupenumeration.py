# setting chirality based on https://www.valencekjell.com/posts/2021-10-15-chiral-templating/index.html

import itertools
import re
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

    def __init__(self, enumerate_kekule = False, permutate = False, insert_small = False):
        """The init."""
        self.multiple = False
        self.columns = ['R1']
        
        self.enumerate_kekule = enumerate_kekule
        self.permutate = permutate
        self.insert_small = insert_small 
        

    def generate_intermediates(self,
                               pair: list[Chem.rdchem.Mol]) -> pd.DataFrame:
        """Method that combines functions in this class to generate intermediates with enumerated r-groups.

        Args:
            pair list[Chem.rdchem.Mol]: pair of structures to generate intermediates for, list of two rdkit molecule objects

        Returns:
            df_interm (pd.DataFrame): dataframe with the generated intermedaites
        """
        self.pair = pair
        self.columns = ['R1']
        self.df = self.get_rgroups(self.enumerate_kekule)
        self.df_interm = self.enumerate_rgroups()
        if re.search('R[2-9]', ' '.join(self.df_interm.keys())):
            self.weld()
            self.clean_intermediates()
        elif self.permutate:
            self.charge_original = Chem.GetFormalCharge(self.pair[0])
            self.permutate_rgroup()
            self.weld()
            self.clean_intermediates()

        #TODO clean up what to return, can also just be a list
        return self.df_interm, self.df_rgroup['Core'][0]


    def get_rgroups(self, enumerate_kekule) -> pd.DataFrame:
        """Method that determines common core and r-groups.

        Use maximum common substructure of two molecules to get the differing r-group(s).
        Sets flag "multiple" to true if there are multiple r-groups.
        Ensures that fused ring r-groups are not present multiple times.

        Returns:
            df_rgroup (pd.DataFrame): dataframe with molecule objects for Core, R1, ... Rn
        """
        res_max = rdFMCS.FindMCS([self.pair[0], self.pair[1]],
                                matchValences=True,
                                ringMatchesRingOnly=True,
                                completeRingsOnly=True,
                                bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact,
                                ringCompare=Chem.rdFMCS.RingCompare.PermissiveRingFusion,
                                timeout=2)
        core_max = Chem.MolFromSmarts(res_max.smartsString)

        # make aromatic bonds explicit
        for i, mol in enumerate(self.pair):
            Chem.Kekulize(mol, clearAromaticFlags=True) # BondType for all modified aromatic bonds will be changed from AROMATIC to SINGLE or DOUBLE

        if enumerate_kekule:
            num_atoms_max = core_max.GetNumAtoms()
            num_atoms = 0

            sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL
            sanitize_flags ^= (Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)

            core = None
            flags = Chem.ResonanceFlags()
            for taut0 in Chem.ResonanceMolSupplier(self.pair[0], flags = flags.KEKULE_ALL):
                Chem.SanitizeMol(taut0, sanitize_flags)
                for taut1 in Chem.ResonanceMolSupplier(self.pair[1], flags = flags.KEKULE_ALL):
                    Chem.SanitizeMol(taut1, sanitize_flags)
                    res_temp = rdFMCS.FindMCS([taut0, taut1],
                                        matchValences=True,
                                        ringMatchesRingOnly=True,
                                        completeRingsOnly=True,
                                        # matchChiralTag=True,
                                        ringCompare=Chem.rdFMCS.RingCompare.PermissiveRingFusion,
                                        timeout=2)
                    core_temp = Chem.MolFromSmarts(res_temp.smartsString)
                    if core_temp.GetNumAtoms() == num_atoms_max:
                        self.pair[0] = taut0
                        self.pair[1] = taut1
                        core = core_temp
                        break
                    elif core_temp.GetNumAtoms() > num_atoms:
                        self.pair[0] = taut0
                        self.pair[1] = taut1
                        core = core_temp
                        num_atoms = core_temp.GetNumAtoms()
            if core is None:
                print('max substructure match not found')
        else:
            smarts = Chem.MolToSmiles(core_max).replace(':', '@').replace('H', '')
            core = Chem.MolFromSmarts(smarts)

        # save molecule information on
        for i, mol in enumerate(self.pair):
            for atom in mol.GetAtoms():
                atom.SetIntProp("SourceAtomIdx", atom.GetIdx())
                atom.SetIntProp("SourceMol", i)

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
            if isinstance(match, tuple):
                self.core_length = len(match[0])
                highest_idx = len(match[0])
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
        if self.df.at[0, 'Core'].HasSubstructMatch(self.df.at[1, 'Core'],useChirality=True):
            combinations = list(
                product(*tuple([self.df[column]
                                for column in self.df.columns[1:]])))
            # create new dataframe for storing adapted rgroups [has Core, R1, ... Rn]
            self.df_interm = pd.DataFrame(combinations,
                                        columns=self.df.columns[1:])
            self.df_interm['Core'] = self.df.at[0, 'Core']
        else:
            combinations = list(
                product(*tuple([self.df[column]
                                for column in self.df.columns])))  
            # create new dataframe for storing adapted rgroups [has Core, R1, ... Rn]
            self.df_interm = pd.DataFrame(combinations,
                                        columns=self.df.columns)

        return self.df_interm
    
    def tokenize(self, rgroup):
        """ 
        Tokenize SMILES of the small and large R group
        """
        # set regex for SMILES 
        pattern =  r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(rgroup)]
        return tokens
    
    def check_charge(self, interm):
        charge = Chem.GetFormalCharge(interm)

        return charge == self.charge_original
    

    # def return_charge(self, rgroup):
    #     try:
    #         mol = pybel.readstring("smi", rgroup)
    #         #bool polaronly, bool correctForPH, double pH
    #         mol.OBMol.AddHydrogens(False, True, 7.4)
    #         rgroup = mol.write("smi")

    #         mol = Chem.MolFromSmiles(rgroup)
    #         charge = Chem.GetFormalCharge(mol)
    #         # - for neutral, positive numbers for positive charge, negative for neragive charge
    #     except:
    #         # in case where pybel cannot read molecule, decided to kick out molecule
    #         charge = None
    #     return charge
    

    def permutate_rgroup(self, column='R1'):
        """ 
        Remove tokens or edit tokens from R-groups of the largest molecule. Currently only able to remove tokens
        """
        self.df_interm_rgroup = pd.DataFrame(columns = self.df.columns[1:])

        tokens = []
        for index, row in self.df_interm.iterrows():
            mol = Chem.Mol(row[column])
            token = self.tokenize(Chem.MolToSmiles(mol))
            tokens.append(token)
            # Chem.SanitizeMol(mol)
            # token = self.tokenize(Chem.MolToSmiles(mol).upper())
            # tokens.append(token)
        
        short_tokens = min(tokens, key=len)
        long_tokens = max(tokens, key=len)
        # sample some/all options where tokens from small r-group are inserted
        available_small = [item for item in short_tokens if not re.match(r"(\[\*\:.\]|\.)", item)]

        if len(available_small) == 0:
            insert_small = False 
        else:
            to_add = set()
            for i in range(1, len(available_small)+1):
                for subset in itertools.combinations(available_small, i):
                    to_add.add(subset)
            insert_small = self.insert_small

        # get all the possible options for shorter r-group   
        # for large rgroup go over all the combinations with length in range 1 shorter than largest fragment - to 1 larger than shortest fragment 
        ## ask willem if shortest intermediate fragment can have as much atoms as shortest fragment or should always be 1 bigger
        ## maybe handle connection token differently
        for i in range(len(long_tokens) - 1, len(short_tokens) - 1, -1):
            for subset in itertools.combinations(long_tokens, i):
                # in some cases connection token will be removed, discard those cases
                ## does not take into account situation with multiple connections in rgroup, like for pair 7 
                ## C1CC1[*:1].[H][*:1].[H][*:1]
                connection = [item for item in subset if re.match('\\[\\*:.\\]', item)]
                if connection:
                    # add fragments of small subset into large subset
                    subsets = []
                    subsets.append(subset)

                    if insert_small:
                        for to_insert in to_add:
                            # only insert tokens from small fragment when smaller than long fragment
                            if len(subset) >= len(tokens) - len(to_insert): continue
                            for j in range(len(subset)):
                                a = list(subset)
                                a[j:j] = to_insert
                                subsets.append(a)

                    for subset in subsets:
                        interm = ''.join(subset)
                        # keep fragments with valid SMILES
                        interm_mol = Chem.MolFromSmiles(interm)
                        if interm_mol is not None:
                            Chem.Kekulize(interm_mol, clearAromaticFlags=True)
                            # keep fragments that do not introduce/loose charge
                            # using openbabel & looking at disconnected rgroups could sometimes be incorrect
                            if self.check_charge(interm_mol):
                                self.df_interm_rgroup.loc[self.df_interm_rgroup.shape[0], column] = interm_mol
        
        # drop duplicate R groups to save time
        self.df_interm_rgroup = self.df_interm_rgroup.drop_duplicates(subset=column)   
        self.df_interm_rgroup['Core'] = self.df.at[0,'Core']
        # in case of multiple rgroups also add unchanged rgroups to df
        # if self.multiple == True:
        #     for rgroup in self.columns:
        #         if rgroup != self.column:
        #             self.df_interm_rgroup[rgroup] = self.df.at[0,rgroup]
        self.df_interm = pd.concat([self.df_interm, self.df_interm_rgroup]).reset_index()


    def weld(self):
        """Put modified rgroups back on the core.
        
        Returns:
            df_interm (pd.DataFrame): now with column Intermediate, contains SMILES of the put together intermediate
        """
        #TODO maybe just return those as a list
        self.df_interm['Intermediate'] = None
        for index, row in self.df_interm.iterrows():
            # if index != len(self.df_interm)-1: continue
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
                numHs = 3 - len(nonHs) - i + atom.GetFormalCharge()
                if numHs < 0:
                    return None
                atom.SetNumExplicitHs(numHs)
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
                numHs = 4 - len(nonHs) - i + atom.GetFormalCharge()
                if numHs < 0:
                    return None
                atom.SetNumExplicitHs(numHs)
            if atom.GetAtomicNum(
            ) == 6 and atom.IsInRing():  # for carbon atoms
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
                numHs = 4 - len(nonHs) - i + atom.GetFormalCharge()
                if numHs < 0:
                    return None
                atom.SetNumExplicitHs(numHs)
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
        try:
            final_mol = Chem.RemoveHs(combined_mol)
        except Chem.rdchem.AtomValenceException:
            final_mol = Chem.RemoveHs(combined_mol, sanitize=False)
            print(final_mol)
        except Chem.rdchem.KekulizeException:
            return None

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
            
        if len(self.df_interm) > 0:
            # remove duplicate molecules
            self.df_interm['SMILES'] = self.df_interm.apply(
                lambda row: Chem.MolToSmiles(row.Intermediate), axis=1)
            self.df_interm = self.df_interm.drop_duplicates(subset=['SMILES'])

        self.df_interm = self.df_interm.drop(
            columns=[*['Core'], *self.columns])
        self.df_interm = self.df_interm.dropna()
