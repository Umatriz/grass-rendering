struct VertexInput {
    float3 pos;
    float3 color;
};

struct VertexOutput {
    float3 color;
    float4 position : SV_Position;
};

struct ViewUniform {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
ConstantBuffer<ViewUniform> ubo;

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;

    output.color = input.color;
    output.position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.pos, 1.0))));

    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target {
    float3 color = inVert.color;
    return float4(color, 1.0);
}

