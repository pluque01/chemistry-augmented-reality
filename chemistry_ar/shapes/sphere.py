from moderngl_window import geometry
import numpy as np


class Sphere:
    def __init__(self, ctx, radius, position, color, projection_matrix):
        self.program = ctx.program(
            vertex_shader="""
            #version 330

            uniform mat4 m_view;
            uniform mat4 m_proj;

            in vec3 in_position;
            in vec3 in_normal;

            out vec3 pos;
            out vec3 normal;

            void main() {
                vec4 VxM = m_view * vec4(in_position, 1.0);
                gl_Position =  m_proj * VxM;
                mat3 m_normal = inverse(transpose(mat3(m_view)));
                normal = m_normal * normalize(in_normal);
                pos = VxM.xyz;
            }
            """,
            fragment_shader="""
            #version 330

            uniform vec4 color;

            in vec3 pos;
            in vec3 normal;

            out vec4 fragColor;

            void main() {
                float l = dot(normalize(-pos), normalize(normal));
                fragColor = color * (0.25 + abs(l) * 0.75);
            }
            """,
        )

        self.vao = geometry.sphere(radius=radius, sectors=32, rings=16)

        self.program["color"] = np.append(color, 1.0)
        self.projection = projection_matrix
        self.position = position

    def render(self, view_matrix):
        # sys.stdout.write(f"X: {pos_x}, Y: {pos_y}, Distance: {distance}\r")
        # sys.stdout.flush()
        self.program["m_proj"].write(self.projection.astype("f4").tobytes())
        # self.program["m_pos"].write(self.position.astype("f4").tobytes())

        # trans = Matrix44.from_translation((20, 20, -20), dtype="f4")
        # rot = Matrix44.from_eulers((time / 2, time / 12.33, time / 11.94), dtype="f4")
        # matrix = rot @ trans
        self.program["m_view"].write(view_matrix.astype("f4").tobytes())
        self.vao.__subclasshook__

        # self.vao.render(moderngl.TRIANGLE_STRIP)

        self.vao.render(self.program)
