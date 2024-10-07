#! /usr/bin/python
import collections
from collections import Counter
from itertools import chain, compress


def copy_directive(line, in_f, out_file):
    """Copies all lines lines of directive without introducting changes"""
    out_file.write(line + "\n")
    while True:
        line = in_f.readline().strip()
        if line == "":
            out_file.write("\n")
            break
        elif line.split()[0] == ";":
            out_file.write(line + "\n")
        else:
            out_file.write(line + "\n")
    return


def process_atoms(line, in_f, out_file, dummy_prefix:str = 'DUM_'):
    """Copies all lines lines of [ atoms ] directive without introducting changes. Creates dictionary of dummy atoms in endstate A and B

    Parameters
    ----------
    dummy_prefix: str
        prefix to distinguish residue names of dummy atoms from physical ones
    
    Returns
    -------
    dict
        dictionary of atom numbers of dummy atoms in endstate A and B
    """
    dummies = {'A': [], 'B': []}
    out_file.write(line  + "\n")

    while True:
        line = in_f.readline()
        if line.strip() == "":
            out_file.write("\n")
            break
        elif line.split()[0] == ";":
            out_file.write(line)
        else:
            nr, typ, resnr, res, atom, cgnr, charge, mass, typB, chargeB, massB = line.strip().split()
            if typ.startswith(dummy_prefix):
                dummies['A'].append(nr)
            if typB.startswith(dummy_prefix):
                dummies['B'].append(nr)
            out_file.write(line)

    return dummies


def process_bonds(line, in_f, out_file, dummies, line_num):
    """Copies all lines lines of [ bonds ] directive without introducting changes. Used to get bridge atoms dummy and physical bridge atom (D&P1), groups bound to P1 (P2), and groups bound to those groups (P3)
           C                
          /                 
       C=C                  
      /   \                 
     D     C 
    
             Physical(P)1─P2-P3
            /              \   
    Dummy(D)                P3 

    Parameters
    ----------
    dummies: dict
        dictionary of atom numbers of dummy atoms in endstate A and B
    line_num: int
        first line of [ bonds ] directive
    
    Returns
    -------
    bridge: dict
        dictionary of bridge in endstate A and B, idx 0 = P1, idx 1 = D
    P2s: dict
        dictionary of atoms connected to P1, key = atom idx P1, values = atom idxs of connected P2 atoms
    P3s: dict
        dictionary of atoms connected to P2, key = atom idx P2, values = atom idxs of connected P3 atoms
    """
    bridge = {'A': [], 'B': []}
    physical = []
    P2s = collections.defaultdict(list)
    P3s = collections.defaultdict(list)
    out_file.write(line + "\n")
    # iterate over bonds to find bridge atoms
    while True:
        line = in_f.readline()
        if line.strip() == "":
            break
        elif line.split()[0] == ";":
            out_file.write(line)
        else:
            ai, aj, funct, c1, k1, c2, k2 = line.strip().split()[:7]
            # bridge is dummy connected to a physical atom
            if ai in dummies['A'] and aj not in dummies['A']:
                bridge['A'].append([aj, ai])  #physical, dummy
                physical.append(aj)
            elif ai not in dummies['A'] and aj in dummies['A']:
                bridge['A'].append([ai, aj])  #physical, dummy
                physical.append(ai)
            if ai in dummies['B'] and aj not in dummies['B']:
                bridge['B'].append([aj, ai])  #physical, dummy
            elif ai not in dummies['B'] and aj in dummies['B']:
                bridge['B'].append([ai, aj])  #physical, dummy
            out_file.write(line)
    out_file.write("\n")

    # iterate over bonds to get P2 atoms
    in_f.seek(line_num)
    while True:
        line = in_f.readline()
        if line.strip() == "":
            break
        elif line.split()[0] == ";":
            continue
        else:
            ai, aj, funct, c1, k1, c2, k2 = line.strip().split()[:7]
            if ai in physical:
                if aj not in dummies['A'] and aj not in dummies['B']:
                    P2s[ai].append(aj)
            if aj in physical:
                if ai not in dummies['A'] and ai not in dummies['B']:
                    P2s[aj].append(ai)

    # iterate over bonds to get P3 atoms
    in_f.seek(line_num)
    P2s_all = [item for row in P2s.values() for item in row]
    while True:
        line = in_f.readline()
        if line.strip() == "":
            break
        elif line.split()[0] == ";":
            continue
        else:
            ai, aj, funct, c1, k1, c2, k2 = line.strip().split()[:7]
            if aj in P2s_all and ai not in P2s.keys(): # if attached to P2s but not a P1 atom
                P3s[aj].append(ai)
            if ai in P2s_all and aj not in P2s.keys(): # if attached to P2s but not a P1 atom
                P3s[ai].append(aj)

    return bridge, P2s, P3s



def process_angles(line,
                   in_f,
                   out_file,
                   bridge_atoms,
                   P2s,
                   fc:str='20.92',
                   multiplicity_atoms=None):
    """Alters angles in [ angles ] directive that anchor dummy atom to physical molecule 
    Implemented as described in DOI: 10.1021/acs.jctc.0c01328 
    Supports dual and triple junctions
    Decoupling of dual junction can be done via: 
    - two 90 degree angles with high force constant (default); or
    - 1 angle 1 dihedral (requires multiplicity_atoms to be set & manual step for dihedral angle)
    For triple junction the force constant of the 3 angles anchoring the dummy atom is lowered

    Parameters
    ----------
    bridge_atoms: dict
        dictionary of bridge in endstate A and B, idx 0 = P1, idx 1 = D
    P2s: dict
        dictionary of atoms connected to P1, key = atom idx P1, values = atom idxs of connected P2 atoms
    fc: str
        value of force constant to use for triple junction, default = 20.92 kJ/mol (5 kcal/mol)
    multiplicity_atoms: optional
        atom idx to keep when anchoring dual junction with 1 angle 1 dihedral (requires multiplicity_atoms to be set)

    """
    out_file.write(line + "\n")
    dual_angle = '90.0'  #degree
    dual_fc = '627.6'  #kJoule mol-1 rad-2
    while True:
        line = in_f.readline()
        if line.strip() == "":
            break
        elif line.split()[0] == ";":
            out_file.write(line)
        else:
            ai, aj, ak, funct, c1, k1, c2, k2 = line.strip().split()[:8]
            comment = " ".join(line.split()[8:])
            atoms = [ai, aj, ak]
            for endstate in ['A', 'B']:
                for physical, dummy in bridge_atoms[endstate]:
                    if physical in atoms and dummy in atoms:
                        if any(ext in atoms
                               for ext in P2s[physical]):
                            # for triple junction set force constant of angle low
                            if len(P2s[physical]) == 3:
                                if endstate == 'A':
                                    line = ai.rjust(6) + aj.rjust(
                                        7) + ak.rjust(6) + funct.rjust(
                                            8) + c1.rjust(15) + fc.rjust(
                                                10) + c2.rjust(20) + k2.rjust(
                                                    15) + comment.rjust(14) + "\n"
                                elif endstate == 'B':
                                    line = ai.rjust(6) + aj.rjust(
                                        7) + ak.rjust(6) + funct.rjust(
                                            8) + c1.rjust(15) + k1.rjust(
                                                15) + c2.rjust(15) + fc.rjust(
                                                    10) + comment.rjust(22) + "\n"
                            # for dual junction
                            elif len(P2s[physical]) == 2:
                                # default option, set force constant high and angle of 90 degrees
                                if multiplicity_atoms is None:
                                    if endstate == 'A':
                                        line = ai.rjust(6) + aj.rjust(7) + ak.rjust(
                                            6) + funct.rjust(8) + dual_angle.rjust(
                                                9) + dual_fc.rjust(16) + c2.rjust(
                                                    20) + k2.rjust(
                                                        15) + comment.rjust(
                                                            13) + "\n"
                                    elif endstate == 'B':
                                        line = ai.rjust(6) + aj.rjust(
                                            7) + ak.rjust(6) + funct.rjust(
                                                8) + c1.rjust(15) + k1.rjust(
                                                    15) + dual_angle.rjust(
                                                        9) + dual_fc.rjust(
                                                            16) + comment.rjust(
                                                            19) + "\n"
                                # second option, remove one angle and keep the other (in multiplicity atoms)
                                else:
                                    if any(set(atoms).issubset(set(ext)) for ext in multiplicity_atoms):  # at idx 2 P2
                                        continue  # if D1, P1, P2 in angle to keep, do not modify
                                    else:
                                        if endstate == 'A':
                                            line = ai.rjust(6) + aj.rjust(
                                                7) + ak.rjust(6) + funct.rjust(
                                                    8) + c1.rjust(15) + "0.000".rjust(
                                                        10) + c2.rjust(20) + k2.rjust(
                                                            15) + comment + "\n"
                                        elif endstate == 'B':
                                            line = ai.rjust(6) + aj.rjust(
                                                7) + ak.rjust(6) + funct.rjust(
                                                    8) + c1.rjust(15) + k1.rjust(
                                                        15) + c2.rjust(15) + "0.000".rjust(
                                                            10) + comment + "\n"
            out_file.write(line)

    out_file.write("\n")
    return


def process_dhedrals(line, in_f, out_file, P2_exceptions, multiplicity_atoms=None):
    """
    Alters dihedrals in [ dihedrals ] directive that couple dummy atom to physical molecule 
    Implemented as described in DOI: 10.1021/acs.jctc.0c01328 
    Remove dihedral angles D1P1P2P3, anchor dihedral D2D1P1P2 via only one of the available P2 atoms:

       \   /                 
        C=C                  
       /   \                 
    D-D   

                             
        P2      P3       
         \     /         
          P1-P2          
         /     \         
    D2-D1       P3 
    
    Parameters
    ----------
    P2_exceptions: list
        P2 atom idxs of for which to not remove the dihedral if in D2D1P1P2, only done for 1 P2 atom per dummy group
    multiplicity_atoms: optional
        atom idx to keep when anchoring dual junction with 1 angle 1 dihedral (requires multiplicity_atoms to be set)
    
    """
    multiplicity_atom_sets = []
    if multiplicity_atoms:
        multiplicity_atom_sets = [set(atoms) for atoms in multiplicity_atoms]
    
    out_file.write(line + "\n")
    while True:
        line = in_f.readline()
        if line.strip() == "":
            out_file.write("\n")
            break
        if line.split()[0] == ";":
            out_file.write(line.strip() + "\n")
        else:
            ai, aj, ak, al, funct, a1, fc1, f1, a2, fc2, f2 = line.strip(
            ).split()[0:11]
            comment = " ".join(line.split()[11:])
            atoms = [ai, aj, ak, al]
            first = line.split()[-1].split('->')[0]
            if first.count("A") > 1 and "D" in first:
                if first.count("D") == 1: # = D1P1P2P3
                    if set(atoms) in multiplicity_atom_sets and fc1 != "0": # for dual anchor when keeping 1 angle, 1 dihedral, increase fc & set multiplicity to 1
                        line = ai.rjust(6) + aj.rjust(7) + ak.rjust(7) + al.rjust(7) + funct.rjust(5) + " " + a1 + " " + "420" + " " + "1" + " " + a2 + " " + fc2 + " " + "1" + " " + comment + "\n"
                    else:
                        line = ai.rjust(6) + aj.rjust(7) + ak.rjust(
                            7
                        ) + al.rjust(7) + funct.rjust(
                            5
                        ) + " " + a1 + " " + "0" + " " + f1 + " " + a2 + " " + fc2 + " " + f2 + " " + comment + "\n"
                elif any(ext in atoms for ext in P2_exceptions): # = D2D1P1P2 & kept
                    if first.count("A") != 2: #unless
                        line = ai.rjust(6) + aj.rjust(7) + ak.rjust(
                            7
                        ) + al.rjust(7) + funct.rjust(
                            5
                        ) + " " + a1 + " " + "0" + " " + f1 + " " + a2 + " " + fc2 + " " + f2 + " " + comment + "\n"
                else: # = D2D1P1P2 & removed
                    line = ai.rjust(6) + aj.rjust(7) + ak.rjust(7) + al.rjust(
                        7
                    ) + funct.rjust(
                        5
                    ) + " " + a1 + " " + "0" + " " + f1 + " " + a2 + " " + fc2 + " " + f2 + " " + comment + "\n"

            second = line.split()[-1].split('->')[1]
            if second.count("A") > 1 and "D" in second:
                if second.count("D") == 1:
                    if set(atoms) in multiplicity_atom_sets and fc2 != "0": # also apply if exact same atoms also have other dihedral term
                        line = ai.rjust(6) + aj.rjust(7) + ak.rjust(7) + al.rjust(7) + funct.rjust(5) + " " + a1 + " " + fc1 + " " + "1" + " " + a2 + " " + "420" + " " + "1" + " " + comment + "\n"
                    else:
                        line = ai.rjust(6) + aj.rjust(7) + ak.rjust(
                            7
                        ) + al.rjust(7) + funct.rjust(
                            5
                        ) + " " + a1 + " " + fc1 + " " + f1 + " " + a2 + " " + "0" + " " + f2 + " " + comment + "\n"
                elif any(ext in atoms for ext in P2_exceptions):
                    if second.count("A") != 2:
                        line = ai.rjust(6) + aj.rjust(7) + ak.rjust(
                            7
                        ) + al.rjust(7) + funct.rjust(
                            5
                        ) + " " + a1 + " " + fc1 + " " + f1 + " " + a2 + " " + "0" + " " + f2 + " " + comment + "\n"
                else:
                    line = ai.rjust(6) + aj.rjust(7) + ak.rjust(7) + al.rjust(
                        7
                    ) + funct.rjust(
                        5
                    ) + " " + a1 + " " + fc1 + " " + f1 + " " + a2 + " " + "0" + " " + f2 + " " + comment + "\n"
            out_file.write(line)
    return


def get_unique_P2s(P2s):
    # get P2s that do not overlap between dummy groups
    bonds_list = [P2s[key] for key in P2s]
    freq = Counter(chain.from_iterable(bonds_list))
    res = {idx for idx in freq if freq[idx] == 1}
    bonds_list_new = [[x for x in bond_list if x in res]
                for bond_list in bonds_list]
    
    return bonds_list_new, res

def get_multiplicity_atoms(bridge, P2s, P3s, res, set_to_90):
    multiplicity_atoms = None
    if not set_to_90:
        # for dual anchored save dihedrals
        multiplicity_atoms = []
        dual_idx = [len(x) == 2 for x in P2s.values()]
        P1s_dual = list(compress(list(P2s.keys()), dual_idx))
        for loc in bridge['A']:
            if loc[0] not in P1s_dual: 
                continue
            P3 = None
            D1_A = loc[1]
            P1 = loc[0]
            P2_atoms = P2s[P1]
            for P2_atom in P2_atoms:
                if P2_atom not in res:
                    continue
                if P2_atom in list(P3s.keys()):
                    P2 = P2_atom
                    P3 = P3s[P2][0]
                    multiplicity_atoms.append((D1_A, P1, P2, P3))
                    D1_B = [
                        x[1] for x in bridge['B']
                        if x[0] == P1
                    ][0]
                    multiplicity_atoms.append((D1_B, P1, P2, P3))
                    break
            if P3 is None:
                for P2_atom in P2_atoms:
                    if P2_atom in list(P3s.keys()):
                        P2 = P2_atom
                        P3 = P3s[P2][0]
                        multiplicity_atoms.append((D1_A, P1, P2, P3))
                        D1_B = [
                            x[1] for x in bridge['B']
                            if x[0] == P1
                        ][0]
                        multiplicity_atoms.append((D1_B, P1, P2, P3))
                        break
            if P3 is None:
                raise KeyError(
                    "No dihedrals found to anchor dual dummy point, use 90 degree setting instead"
                )
            
    return multiplicity_atoms


def process_file(in_file, out_file, set_to_90=True, angle_forceconstant='20.92'):
    with open(out_file, 'w') as out_f:
        with open(in_file) as in_f:
            for line in iter(in_f.readline, ''):
                if line.strip() == "[ moleculetype ]":
                    print("moleculetype section started!")
                    copy_directive(line.strip(), in_f, out_f)
                    print("moleculetype section done!")
                elif line.strip() == "[ atoms ]":
                    print("atoms section started!")
                    dummies = process_atoms(line.strip(), in_f, out_f)
                    print("atoms section done!")
                elif line.strip() == "[ bonds ]":
                    print("bonds sectrion started!")
                    bridge, P2s, P3s = process_bonds(line.strip(), in_f, out_f,
                                                     dummies, in_f.tell())
                    print("bonds sectrion done!")
                elif line.strip() == "[ pairs ]":
                    print("pairs section started!")
                    copy_directive(line.strip(), in_f, out_f)
                    print("pairs section done!")
                elif line.strip() == "[ angles ]":
                    print("angles section started!")
                    # get P2s that do not overlap between attachment point dummies
                    bonds_list, res = get_unique_P2s(P2s)
                    multiplicity_atoms = None
                    if not set_to_90:
                        multiplicity_atoms = get_multiplicity_atoms(bridge, P2s, P3s, res, set_to_90)
                    process_angles(line.strip(),
                                   in_f,
                                   out_f,
                                   bridge_atoms=bridge,
                                   P2s=P2s,
                                   fc=angle_forceconstant,
                                   multiplicity_atoms=multiplicity_atoms)
                    print("angles section done!")
                elif line.strip() == "[ dihedrals ]":
                    print("dihedrals section started!")
                    dihedral_excpt = [suitable[0] for suitable in bonds_list]
                    process_dhedrals(line.strip(),
                                     in_f,
                                     out_f,
                                     P2_exceptions=dihedral_excpt,
                                     multiplicity_atoms=multiplicity_atoms)
                    print("dihedrals section done!")
                elif line.strip() == "[ cmap ]":
                    print("cmap section started!")
                    copy_directive(line.strip(), in_f, out_f)
                    print("cmap section done!")
                elif line.strip() == "":
                    out_f.write(line)
        in_f.close()
    out_f.close()

if __name__ == "__main__":
    process_file('ref_sec.itp', 'ref_sec_decoupled_new_90.itp')
