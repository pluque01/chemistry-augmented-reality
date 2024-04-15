import numpy as np
import moderngl
from pyrr import Matrix44
from moderngl_window import geometry


class Cube:
    def __init__(self, ctx, aspect_ratio):
        self.ctx = ctx
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330

                uniform mat4 MV;
                uniform mat4 P;

                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;

                void main() {
                    gl_Position = P * MV * vec4(in_position, 1.0);
                    v_vert = in_position;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                }
            """,
            fragment_shader="""
                #version 330

                uniform vec3 Light;
                uniform vec4 color = vec4(1.0, 0.0, 0.0, 1.0);

                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;

                out vec4 f_color;

                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    f_color = color * lum;
                }
            """,
        )
        self.vao = geometry.cube(size=(0.5, 0.5, 0.5), center=(0.0, 0.0, 0.0))
        self.projection = Matrix44.perspective_projection(
            45,
            aspect_ratio,
            0.1,
            10,
            dtype="f4",  # FOV, aspect ratio, near, far
        )
        print("Proyección antigua: \n", self.projection)

    def render(self, view_matrix, projection):
        # self.program["projection"].write(self.projection)
        # self.program["model"].write(view_matrix.astype("f4").tobytes())
        self.program["MV"].write(view_matrix.astype("f4").tobytes())
        self.program["P"].write(projection.astype("f4").tobytes())
        self.vao.__subclasshook__
        self.vao.render(self.program)
