#version 450

layout(location = 0) in vec3 VertexPosition;
layout(location = 1) in vec3 VertexColor;

layout(location = 0) out vec3 ColorOut;

void main()
{
    gl_Position = vec4(VertexPosition, 1.0);

    ColorOut = VertexColor;
}