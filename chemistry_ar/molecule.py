from typing import List, Dict
from shapes.sphere import Sphere

registered_atoms: Dict[str, float] = dict(
    H=0.31,
    He=0.28,
    Li=1.28,
    Be=0.96,
    B=0.84,
    C=0.76,
    N=0.71,
    O=0.66,
    F=0.57,
    Ne=0.58,
    Na=1.66,
    Mg=1.41,
    Al=1.21,
    Si=1.11,
    P=1.07,
    S=1.05,
    Cl=1.02,
    Ar=1.06,
    K=2.03,
    Ca=1.76,
    Sc=1.7,
    Ti=1.6,
    V=1.53,
    Cr=1.39,
    Mn=1.39,
    Fe=1.32,
    Co=1.26,
    Ni=1.24,
    Cu=1.32,
    Zn=1.22,
    Ga=1.22,
    Ge=1.2,
    As=1.19,
    Se=1.2,
    Br=1.2,
    Kr=1.16,
    Rb=2.2,
    Sr=1.95,
    Y=1.9,
    Zr=1.75,
    Nb=1.64,
    Mo=1.54,
    Tc=1.47,
    Ru=1.46,
    Rh=1.42,
    Pd=1.39,
    Ag=1.45,
    Cd=1.44,
    In=1.42,
    Sn=1.39,
    Sb=1.39,
    Te=1.38,
    I=1.39,
    Xe=1.4,
    Cs=2.44,
    Ba=2.15,
    La=2.07,
    Ce=2.04,
    Pr=2.03,
    Nd=2.01,
    Pm=1.99,
    Sm=1.98,
    Eu=1.98,
    Gd=1.96,
    Tb=1.94,
    Dy=1.92,
    Ho=1.92,
    Er=1.89,
    Tm=1.9,
    Yb=1.87,
)


class Atom(Sphere):
    def __init__(self, ctx, name: str, radius: float, projection_matrix):
        super().__init__(ctx, radius, projection_matrix)
        self.name = name

    def render(self, view_matrix):
        print(f"Rendering {self.name}")
        super().render(view_matrix)


class Molecule:
    def __init__(self, ctx, atoms: List[str], aruco: int, projection_matrix):
        self.ctx = ctx
        self.aruco = aruco
        self.projection_matrix = projection_matrix
        self.atoms = []
        for atom in atoms:
            radius = registered_atoms[atom]
            self.atoms.append(Atom(ctx, atom, radius, projection_matrix))

    def render(self, view_matrix):
        for atom in self.atoms:
            atom.render(view_matrix)
