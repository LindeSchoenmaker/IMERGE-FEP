#! /usr/bin/python



def process_moleculetypes(line, in_f, out_file):
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

def process_atoms(line, in_f, out_file):
    out_file.write(line)
    atoms_lig_B = []
    atoms_lig_AB = []
    while True:
        line = in_f.readline()
        if line.strip() == "":
            out_file.write("\n")
            break
        elif line.split()[0] == ";":
            pass
        else:
            nr, typ, resnr, res, atom, cgnr, charge, mass, typB, chargeB, massB = line.strip().split()
            atoms_lig_AB.append(nr.rjust(6) + typ.rjust(12) + resnr.rjust(7) + res.rjust(7) + atom.rjust(7) + cgnr.rjust(7) + charge.rjust(11) + mass.rjust(11) + " " + typB.rjust(11) + '0.000000'.rjust(11) + massB.rjust(11))
            atoms_lig_B.append(nr.rjust(6) + typB.rjust(12) + resnr.rjust(7) + res.rjust(7) + atom.rjust(7) + cgnr.rjust(7) + '0.000000'.rjust(11) + massB.rjust(11) + " " + typB.rjust(11) + chargeB.rjust(11) + massB.rjust(11))
    out_file.write(";    nr           type resnr residue atom  cgnr     charge    mass          typeB    chargeB   massB\n")

    out_file.write("#ifdef LIGAND_AB\n")
    for line in atoms_lig_AB:
        out_file.write(line + "\n")
    out_file.write("#endif\n\n")
    out_file.write("#ifdef LIGAND_B\n")
    for line in atoms_lig_B:
        out_file.write(line + "\n")
    out_file.write("#endif\n")
    
    return

def process_bonds(line, in_f, out_file):
    out_file.write(line + "\n")
    last_bonds_section = []
    while True:
        line = in_f.readline()
        if line.strip() == "":
            break
        elif line.split()[0] == ";":
            out_file.write(line)
        else:
            ai, aj, funct, c1, k1, c2, k2 = line.strip().split()[:7]
            if c1 == c2 and k1 == k2:
                out_file.write(line)
            else:
                last_bonds_section.append(line.strip())

    if len(last_bonds_section) != 0:
        out_file.write("#ifdef LIGAND_AB\n")
        for line in last_bonds_section:
            out_file.write("    " + line + "\n")
        out_file.write("#endif\n")

        out_file.write("#ifdef LIGAND_B\n")
        for line in last_bonds_section:
            ai, aj, funct, c1, k1, c2, k2 = line.strip().split()[:7]
            out_file.write("    " + ai + "    " + aj + "\t" + funct + "\t" +  c2 + "\t" + k2 + "\t" + c2 + "\t" + k2 + "\n")
        out_file.write("#endif\n")
        
    out_file.write("\n\n")
    return

def process_pairs(line, in_f, out_file):
    out_file.write(line + "\n")
    last_pairs_section = []
    while True:
        line = in_f.readline()
        if line.strip() == "":
            out_file.write("\n\n")
            break
        elif line.split()[0] == ";":
            out_file.write(line)
        else:
            if len(line.split()) == 7:
                ai, aj, funct, c0, c1, c2, c3 = line.strip().split()[:7]
                if c0 == c2 and c1 == c3:
                    out_file.write(line)
                else:
                    last_pairs_section.append(line.strip())
            else:
                out_file.write(line)
    if len(last_pairs_section) != 0:
        out_file.write("#ifdef LIGAND_AB\n")
        for line in last_pairs_section:
            out_file.write("    " + line + "\n")
        out_file.write("#endif\n")
        
        out_file.write("#ifdef LIGAND_B\n")
        for line in last_pairs_section:
            ai, aj, funct, c0, c1, c2, c3 = line.strip().split()[:7]
            out_file.write("    " + ai + "    " + aj + "\t" + funct + "\t" + c2 + "  " + c3 + "\t" + c2 + "  " + c3 + "\n")
        out_file.write("#endif\n")

        
    out_file.write("\n\n")

    return

def process_angles(line, in_f, out_file):
    out_file.write(line + "\n")
    last_angless_section = []
    while True:
        line = in_f.readline()
        if line.strip() == "":
            break
        elif line.split()[0] == ";":
            out_file.write(line)
        else:
            ai, aj, ak, funct, c1, k1, c2, k2 = line.strip().split()[:8]
            if c1 == c2 and k1 == k2:
                out_file.write(line)
            else:
                last_angless_section.append(line.strip())
    if len(last_angless_section) != 0:
        out_file.write("#ifdef LIGAND_AB\n")
        for line in last_angless_section:
            out_file.write("    " + line + "\n")
        out_file.write("#endif\n")

        out_file.write("#ifdef LIGAND_B\n")
        for line in last_angless_section:
            ai, aj, ak, funct, c1, k1, c2, k2 = line.strip().split()[:8]
            out_file.write("    " + ai + "    " + aj + "    " + ak + "    " + funct + "\t" +  c2 + "\t" + k2 + "\t" + c2 + "\t" + k2 + "\n")
        out_file.write("#endif\n")

        
    out_file.write("\n\n")
    return

def process_dhedrals(line, in_f, out_file):
    out_file.write(line + "\n")
    dihedrals_lig_B = []
    dihedrals_lig_AB = []
    while True:
        line = in_f.readline()
        if line.strip() == "":
            out_file.write("\n")
            break
        if line.split()[0] == ";":
            out_file.write(line.strip())
        else:
            ai, aj, ak, al, funct, a1, fc1, f1, a2, fc2, f2 = line.strip().split()[0:11]
            comment = " ".join(line.split()[11:])
            dihedrals_lig_B.append(ai.rjust(6) + aj.rjust(7) + ak.rjust(7) + al.rjust(7) + funct.rjust(5) + " " + a2 + " " + fc2 + " " + f2 + " " + a2 + " " + fc2 + " " + f2 + " " + comment)
            dihedrals_lig_AB.append(line.strip())
    out_file.write("#ifdef LIGAND_AB\n")
    for line in dihedrals_lig_AB:
        out_file.write(line + "\n")
    out_file.write("#endif\n\n")
    out_file.write("#ifdef LIGAND_B\n")
    for line in dihedrals_lig_B:
        out_file.write(line + "\n")
    out_file.write("#endif\n")
    return

def process_cmap(line, in_f, out_file):
    out_file.write(line + "\n")
    while True:
        line = in_f.readline()
        if line.strip() == "":
            out_file.write("\n")
            break
        elif line.split()[0] == ";":
            out_file.write(line)
        else:
            out_file.write(line)
    return

def process_file(in_file, out_file):
    with open(out_file, 'w') as out_f:
        with open(in_file) as in_f:
            for line in in_f:
                if line.strip() == "[ moleculetype ]":
                    print("moleculetype section started!")
                    process_moleculetypes(line.strip(), in_f, out_f)
                    print("moleculetype section done!")
                elif line.strip() == "[ atoms ]":
                    print("atoms section started!")
                    process_atoms(line.strip(), in_f, out_f)
                    print("atoms section done!")
                elif line.strip() == "[ bonds ]":
                    print("bonds sectrion started!")
                    process_bonds(line.strip(), in_f, out_f)
                    print("bonds sectrion done!")
                elif line.strip() == "[ pairs ]":
                    print("pairs section started!")
                    process_pairs(line.strip(), in_f, out_f)
                    print("pairs section done!")
                elif line.strip() == "[ angles ]":
                    print("angles section started!")
                    process_angles(line.strip(), in_f, out_f)
                    print("angles section done!")
                elif line.strip() == "[ dihedrals ]":
                    print("dihedrals section started!")
                    process_dhedrals(line.strip(), in_f, out_f)
                    print("dihedrals section done!")
                elif line.strip() == "[ cmap ]":
                    print("cmap section started!")
                    process_cmap(line.strip(), in_f, out_f)
                    print("cmap section done!")
                elif line.strip() == "":
                    out_f.write(line)
        in_f.close()
    out_f.close()

