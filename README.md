# Chemistry-AR

Chemistry-AR is an augmented reality application that allows you to visualize and interact with chemical molecules using ArUco markers.

## Features

- Real-time detection of markers using OpenCV and ArUco markers
- Rendering of 3D molecules using OpenGL through [ModernGL](https://github.com/moderngl/moderngl)
- Text overlay of molecule names on markers
- Support for multiple markers and molecules simultaneously

### Planned features

- Combine atoms to form your own molecules
- Adapt the content based on the user experience

## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/pluque01/chemistry-ar.git
    ```

2. Install the required dependencies:

    ```shell
    poetry install
    ```

3. Run the application:

    ```shell
    poetry run chemistry-ar/chemistry-ar.py
    ```

## Usage

1. Print the ArUco markers provided in the `markers` directory.

2. Launch the application and point your webcam towards the printed markers.

3. The application will detect the markers and render the corresponding molecules in real-time.

4. Interact with the molecules by moving the markers.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).