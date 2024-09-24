EV2kJPerMol = 0.073498618
Hatree2kJPerMol = 2625.499638
Hatree2kCalPerMol = 627.509474
kJ2kCal = 0.239005736
kCal2kJ = 4.184


sugar_angle_v3 = {
    "C3'-endo": -35.70,
    "C2'-endo": 23.15,
}

## dihedral angle constraints
# O4'-C4'-C3'-C2' to -35.70 degree
# C4'-C3'-O3'-H3T to -120 degree
# C3'-C2'-O2'-HO2' to -120 degree
# C4'-C5'-O5'-HO5' to 180 degree
dihedral_constraint_templates = [
    ["O4'", "C4'", "C3'", "C2'", sugar_angle_v3["C3'-endo"]],
    ["C4'", "C3'", "O3'", "HO3'", -120.0],
    ["C3'", "C2'", "O2'", "HO2'", -120.0],
    ["C4'", "C5'", "O5'", "HO5'", 180.0]
]


### 
# Pseudo dihedral angle constraints
# Reference: https://pubs.acs.org/doi/10.1021/ct401013s
# Zx = v1+v3/(2*cos(4pi/5))
# Zy = v1-v3/(2*sin(4pi/5))
# Pseudoroation angle: P = arctan(Zy/Zx)
# For C3'-endo, P = 18 drgree

DIHDEDRAL_CONSTRAINTS = {
    "C3'-endo-constraint-v1": {
        "atoms": ["O4'", "C1'", "C2'", "C3'"],
        "angle": -22.2
    },
    "C3'-endo-constraint-v3": {
        "atoms": ["O4'", "C4'", "C3'", "C2'"],
        "angle": -36.0
    },
    "O3'-constraint": {
        "atoms": ["C4'", "C3'", "O3'", "HO3'"],
        "angle": -120.0
    },
    "O2'-constraint": {
        "atoms": ["C3'", "C2'", "O2'", "HO2'"],
        "angle": -120.0
    },
    "O5'-constraint": {
        "atoms": ["C4'", "C5'", "O5'", "HO5'"],
        "angle": 180.0
    }
}

DIHDEDRAL_CONSTRAINTS_PHOSPHATE = {
    "O2'-constraint": {
        "atoms": ["C3'", "C2'", "O2'", "HO2'"],
        "angle": 150.0
    },
    "alpha": {
        "atoms": ["O3'", "P", "O52", "C01"],
        "angle": -66.0
    },
}


PSEUDOROTATION = {
    "C3'-endo": {
        "phase": 15.43,
        "intensity": 0.6760
    },
    
    "C2'-endo": {
        "phase": -16.67,
        "intensity": 0.6706
    },
}

CHI_ANGLE = {
    "syn": 55.41,
    "anti": 203.56-360.0  # -155.79
}

