import numpy as np
import moderngl


class Rectangle:
    def __init__(self, ctx, width, height):
        self.texture = ctx.texture((width, height), 3)
        self.program = ctx.program(
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
        # Define el vértices y coordenadas de textura del rectángulo
        self.vertices = np.array(
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

        # Define los índices del rectángulo
        # self.indices = np.array([0, 1, 2, 3], dtype="i4")

        self.vbo = ctx.buffer(self.vertices)
        # self.ibo = ctx.buffer(self.indices)

        self.texture = ctx.texture((width, height), 3)

        self.vao = ctx.vertex_array(
            self.program,
            [
                (self.vbo, "2f", "in_vert"),
            ],
        )

    def render(self, texture: bytes):
        # Actualiza la textura del rectángulo con el nuevo fotograma
        self.texture.write(texture)
        self.texture.use()

        self.vao.render(moderngl.TRIANGLE_STRIP)
