struct VertexInput {
    float3 pos;
    float3 color;
    float3 normal;
    float2 uv;
};

struct VertexOutput {
    float3 color;
    float4 sv_position : SV_Position;
    float3 position;
    float3 normal;
    float2 uv;
};

struct ViewUniform {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float3x3 normal;
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
    output.normal = mul(ubo.normal, input.normal);
    output.uv = input.uv;

    return output;
}

Sampler2D texture;

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

    float3 out_color = (ambient + diffuse) * texture.Sample(input.uv).rgb;
    return float4(out_color, 1.0);
}

