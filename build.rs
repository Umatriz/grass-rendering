use std::{
    io,
    process::{Command, Output},
};

fn main() {
    println!("cargo::rerun-if-changed=shaders/triangle.slang");
    // slangc shader.slang -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name -entry vertMain -entry fragMain -o slang.spv
    compile_slang("shaders/triangle.slang", &["vertMain", "fragMain"]).unwrap();
}

fn compile_slang(input: &str, entries: &[&str]) -> io::Result<Output> {
    let mut output = input.strip_suffix(".slang").map(|s| s.to_string()).unwrap();
    output.push_str(".spv");

    Command::new("slangc")
        .arg(input)
        .args([
            "-target",
            "spirv",
            "-profile",
            "spirv_1_4",
            "-emit-spirv-directly",
            "-fvk-use-entrypoint-name",
        ])
        .args(entries.iter().flat_map(|entry| ["-entry", entry]))
        .arg("-o")
        .arg(output)
        .output()
}
