import moderngl
from PIL import Image
from OpenGL import GL
import numpy as np
import cv2
from threading import Thread, Event


from _window import Window

thread_quit = Event()
cap = cv2.VideoCapture(cv2.CAP_DSHOW)
new_frame = cap.read()[1]


def init_video():
    print("Initializing video...")
    video_thread = Thread(target=update_video, args=())
    video_thread.start()


def update_video():
    global new_frame
    while not thread_quit.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from video capture")
            break
        new_frame = frame
    cap.release()
    cv2.destroyAllWindows()


class ChemistryAR(Window):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.texture = None

        self.prog = self.ctx.program(
            # Creamos el shader para dibujar la imagen
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                out vec2 frag_texcoord;
                void main() {
                    frag_texcoord = in_vert * 0.5 + 0.5;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D texture;
                in vec2 frag_texcoord;
                out vec4 fragColor;
                void main() {
                    fragColor = texture2D(texture, frag_texcoord);
                }
            """,
        )
        # Creamos los vértices de un rectángulo que cubra toda la pantalla
        vertices = np.array(
            [
                -1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
            ],
            dtype="f4",
        )

        # Creamos el VAO (Vertex Array Object) y el VBO (Vertex Buffer Object)
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_vert")])

    def render(self, time, frame_time):
        global new_frame

        frame = new_frame
        if self.texture:
            self.texture.release()
        # convert image to OpenGL texture format
        tx_image = cv2.flip(frame, 0)
        tx_image = Image.fromarray(tx_image)
        tx_size = (tx_image.size[0], tx_image.size[1])
        tx_image = tx_image.tobytes("raw", "BGRX", 0, 1)
        texture = GL.glGenTextures(1)
        # GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            tx_size[0],
            tx_size[1],
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            tx_image,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        self.texture = self.ctx.external_texture(texture, tx_size, 4, 0, "f1")

        self.ctx.clear()
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.texture.use()

        self.vao.render(moderngl.TRIANGLE_STRIP)

    def close(self):
        thread_quit.set()
        super().close()


if __name__ == "__main__":
    init_video()
    ChemistryAR.run()
