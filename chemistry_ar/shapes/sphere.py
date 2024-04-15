from moderngl_window import geometry


class Sphere:
    def __init__(self, ctx, radius, projection_matrix):
        self.program = ctx.program(
            vertex_shader="""
            #version 330

            uniform mat4 m_model;
            uniform mat4 m_proj;

            in vec3 in_position;
            in vec3 in_normal;

            out vec3 pos;
            out vec3 normal;

            void main() {
                vec4 p = m_model * vec4(in_position, 1.0);
                gl_Position =  m_proj * p;
                mat3 m_normal = inverse(transpose(mat3(m_model)));
                normal = m_normal * normalize(in_normal);
                pos = p.xyz;
            }
            """,
            fragment_shader="""
            #version 330

            out vec4 fragColor;
            uniform vec4 color = vec4(1.0, 0.0, 0.0, 1.0);

            in vec3 pos;
            in vec3 normal;

            void main() {
                float l = dot(normalize(-pos), normalize(normal));
                fragColor = color * (0.25 + abs(l) * 0.75);
            }
            """,
            # fragment_shader="""
            # #version 330
            #
            # in vec3 FragPos; // Posición del fragmento en coordenadas del mundo
            # in vec3 normal; // Normal del fragmento
            #
            # out vec4 fragColor;
            #
            # void main()
            # {
            #     // normalizamos la normal
            #     vec3 n = normalize(normal);
            #
            #     // Calculamos un factor que determina la mezcla de color entre rojo y azul
            #     float mixFactor = (n.z + 1.0) / 2.0;
            #
            #     // Creamos el color mezclado entre rojo y azul
            #     vec3 mixedColor = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), mixFactor);
            #
            #     fragColor = vec4(mixedColor, 1.0);
            # }
            # """,
        )

        self.vao = geometry.sphere(radius=radius, sectors=32, rings=16)

        self.projection = projection_matrix

    def render(self, view_matrix):
        # sys.stdout.write(f"X: {pos_x}, Y: {pos_y}, Distance: {distance}\r")
        # sys.stdout.flush()
        self.program["m_proj"].write(self.projection.astype("f4").tobytes())

        # trans = Matrix44.from_translation((20, 20, -20), dtype="f4")
        # rot = Matrix44.from_eulers((time / 2, time / 12.33, time / 11.94), dtype="f4")
        # matrix = rot @ trans
        self.program["m_model"].write(view_matrix.astype("f4").tobytes())
        self.vao.__subclasshook__

        # self.vao.render(moderngl.TRIANGLE_STRIP)

        self.vao.render(self.program)
