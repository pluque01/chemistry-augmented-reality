import face_recognition
import json
import os
from .models import User, UserEncoder


class DatabaseManager:
    def __init__(self, database_path="user_database.json"):
        self.database_path = database_path
        self.users: list[User] = []
        # self.known_face_encodings = []
        # self.known_face_names = []
        self.load_database()

    def load_database(self):
        """Load the user database from the JSON file."""
        if os.path.exists(self.database_path):
            with open(self.database_path, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
                for user in data:
                    self.users.append(
                        User(user["name"], user["face_encoding"], user["level"])
                    )
        else:
            raise FileNotFoundError(f"Database file not found: {self.database_path}")

    def save_database(self):
        """Save the user database to the JSON file."""
        with open(self.database_path, "w") as file:
            json.dump(self.users, file, cls=UserEncoder)

    def add_user(self, user_name, face_encoding) -> User:
        """Add a new user with the given name and enconding."""
        self.users.append(User(user_name, face_encoding, 0))
        self.save_database()
        print(f"User {user_name} added successfully!")
        return self.users[-1]

    def recognize_user(self, image, location=None) -> User | None:
        """Recognize the user in the given image."""
        face_encodings = face_recognition.face_encodings(image, location)
        if not face_encodings:
            print("No face found in the image.")
            return None

        known_face_encodings = [user.face_encoding for user in self.users]
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encodings[0]
        )
        if not any(matches):
            print("User not found.")
            return None
        # TODO: Check if the user is correct with enumerate
        for i, match in enumerate(matches):
            if match:
                return self.users[i]
        return None
