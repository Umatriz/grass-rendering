struct VertexInput {
    float3 pos;
    float3 color;
    float3 normal;
};

struct VertexOutput {
    float3 color;
    float4 sv_position : SV_Position;
    float3 position;
    float3 normal;
};

struct ViewUniform {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float3 eye_pos;
    float3 light_pos;
};
ConstantBuffer<ViewUniform> ubo;

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;

    output.color = input.color;
    output.sv_position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.pos, 1.0))));
    output.position = mul(ubo.model, float4(input.pos, 1.0)).xyz;
    output.normal = input.normal;

    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target {
    float3 light_color = float3(1.0);
    float ambient_strength = 0.1;
    // TODO: multiply with light color
    float3 ambient = ambient_strength * light_color;

    float3 normal = normalize(input.normal);
    float3 light_dir = normalize(ubo.light_pos - input.position);

    float diff = saturate(dot(normal, light_dir));
    float3 diffuse = diff * light_color;

    float3 out_color = (ambient + diffuse) * input.color;
    return float4(out_color, 1.0);
}

