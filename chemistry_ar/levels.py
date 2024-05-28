import toml

levels_data = toml.load("chemistry_ar/data/levels.toml")


class Level:
    def __init__(self, level_number: int):
        self.level_number = level_number
        self.level_markers = self.get_marker_data(
            levels_data["levels"][level_number]["markers"]
        )

    def get_marker_data(self, markers) -> list:
        marker_list = []
        for m in markers:
            marker_list.append(LevelMarker(m["required"], m["atoms"]))
        return marker_list

    def get_level_markers(self):
        return self.level_markers


class LevelMarker:
    def __init__(self, required: bool, atoms: dict):
        self.required = required
        self.atoms = self.get_atoms_data(atoms)

    def get_atoms_data(self, atoms) -> list:
        atom_list = []
        for a in atoms:
            atom_list.append(LevelMarkerAtom(a["element"], a["count"]))
        return atom_list

    def get_name(self) -> str:
        name = ""
        for a in self.atoms:
            name += a.element + str(a.count)
        return name


class LevelMarkerAtom:
    def __init__(self, element: str, count: int):
        self.element = element
        self.count = count
