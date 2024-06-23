import toml
import yaml


class Level:
    def __init__(self, level_data):
        self.level_markers = self.get_marker_data(level_data["markers"])
        self.level_objective = level_data["objective"]
        self.markers_for_solution = 0
        self.number_of_markers = 0
        for m in self.level_markers:
            if m.required:
                self.markers_for_solution += 1
            self.number_of_markers += 1

    def get_marker_data(self, markers) -> list:
        marker_list = []
        for m in markers:
            marker_list.append(LevelMarker(m["required"], m["atoms"]))
        return marker_list

    def get_markers(self):
        return self.level_markers

    def get_objective_name(self):
        return self.level_objective["name"]

    def get_objective_smiles(self):
        return self.level_objective["smiles"]

    def get_objetive_markers_amount(self):
        return self.markers_for_solution

    def get_number_of_markers(self):
        return self.number_of_markers


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


class GameLevels:
    current_level_number: int = 0

    def __init__(self):
        with open("chemistry_ar/data/levels.yaml", "r") as file:
            self.levels_data = yaml.safe_load(file)
        self.list_of_levels: dict[int, Level] = {}
        for i in range(len(self.levels_data["levels"])):
            self.list_of_levels[i] = Level(self.levels_data["levels"][i])

    def get_current_level_number(self) -> int:
        return self.current_level_number

    def get_current_level(self) -> Level:
        return self.list_of_levels[self.current_level_number]

    def set_current_level(self, level_number: int) -> None:
        if level_number >= self.get_number_of_levels():
            raise ValueError("Level number out of bounds")
        self.current_level_number = level_number

    def get_number_of_levels(self) -> int:
        return len(self.list_of_levels)
