struct VertexInput {
    float2 inPos;
    float3 inColor;
};

struct VertexOutput {
    float3 color;
    float4 sv_position : SV_Position;
};

static float2 positions[3] = float2[](
    float2(0.0, -0.5),
    float2(0.5, 0.5),
    float2(-0.5, 0.5)
);

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;
    output.color = input.inColor;
    output.sv_position = float4(input.inPos, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target {
    float3 color = inVert.color;
    return float4(color, 1.0);
}

