# Chemistry-AR

Chemistry-AR is an augmented reality application that allows you to visualize and interact with chemical molecules using ArUco markers. You are given a set of atoms and your objective is to create the specified molecule. 

The documentation in Spanish can be read [here](QuimicAR.pdf).

## Features

- Real-time detection of markers using OpenCV and ArUco markers
- Rendering of 3D molecules using OpenGL through [ModernGL](https://github.com/moderngl/moderngl)
- Recognize user with [face_recognition](https://github.com/ageitgey/face_recognition)
- Voice input with [SpeechRecognition](https://github.com/Uberi/speech_recognition)


## Installation

### Using venv

1. Clone the repository:

   ```shell
   git clone https://github.com/pluque01/chemistry-augmented-reality.git QuimicAR && cd QuimicAR
   ```

1. Create a virtual environment:

   ```shell
   python -m venv venv
   ```

1. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

1. Run the application:

   ```shell
   python chemistry-ar/chemistry-ar.py
   ```

### Using Poetry

1. Clone the repository:

   ```shell
   git clone https://github.com/pluque01/chemistry-augmented-reality.git QuimicAR && cd QuimicAR
   ```

1. Install the required dependencies:

   ```shell
   poetry install
   ```

1. Run the application:

   ```shell
   poetry run chemistry-ar/chemistry-ar.py
   ```

## Usage

> [!IMPORTANT]
> This application uses the dictionary of 6x6 ArUco markers, so be sure to print at least 4 of them.

The first time the application is run, it attempts to identify the user and will ask for a name to register them. The application uses English for communication. Once the user is registered, the first puzzle is loaded. From this point, it is recommended that the camera be pointed at a flat surface with good lighting.

In the user interface, the target molecule for the puzzle appears in the upper left corner, along with the number of markers needed to display all the atoms. Not all atoms will be necessary to solve the puzzle, so the user must identify which atoms will allow them to solve it.

Once the necessary atoms are identified, the user can physically group them by bringing them closer together. The application will detect the grouping, and if it is correct, it will generate the target molecule. If the grouping is incorrect, the application will generate nothing. The resulting molecule can be separated by moving the markers apart.

When the solution is found, the application asks the user if they want to move on to the next puzzle. The user must say "yes" or "no." If the user says "yes," the application will load the next puzzle. If the user says "no," the application will continue displaying the current puzzle. This question will be asked each time the target molecule is combined.

When the user has completed all the puzzles, the application will congratulate them and start over from the first puzzle.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

