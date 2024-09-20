# rhfe_ligands

folders:
- ligands
contains sdf files each with 3 or more molecules with aligned MCS

Molecule naming convention: 
- set that belongs to (number)
- parent (P) or intermediate (I)
- identifier (A/B for parents, number for intermediates)

summary of forcefield settings:
water model: tip3p
    forcefields: "amber/ff14SB.xml", "amber/tip3p_standard.xml", "amber/tip3p_HFE_multivalent.xml", "amber/phosaa10.xml"
small_molecule_forcefield: openff_unconstrained-2.1.1

