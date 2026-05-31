struct VertexInput {
    float3 pos;
    float3 color;
    float3 normal;
};

struct VertexOutput {
    float3 color;
    float4 position : SV_Position;
    float3 normal;
};

struct ViewUniform {
    float4x4 model;
    float4x4 view;
    float4x4 proj;

    float3 light_pos;
};
ConstantBuffer<ViewUniform> ubo;

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;

    output.color = input.color;
    output.position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.pos, 1.0))));
    output.normal = input.normal;

    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target {
    float light_color = saturate(dot(ubo.light_pos, inVert.normal));

    float3 color = float3(light_color);
    return float4(color, 1.0);
}

