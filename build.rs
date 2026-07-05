use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(mut command: Command) {
    let status = command.status().unwrap_or_else(|error| {
        panic!("failed to start {:?}: {error}", command);
    });
    if !status.success() {
        panic!("{:?} failed with status {status}", command);
    }
}

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let runtime_dir = manifest_dir.join("runtime").join("llama.cpp");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
    let build_dir = out_dir.join("funasr-cmake");
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let build_type = if profile == "release" {
        "Release"
    } else {
        "RelWithDebInfo"
    };

    println!("cargo:rerun-if-env-changed=FUNASR_LLAMA_SOURCE");
    println!("cargo:rerun-if-changed=runtime/llama.cpp/CMakeLists.txt");
    println!("cargo:rerun-if-changed=runtime/llama.cpp/funasr-capi/funasr_c_api.h");
    println!("cargo:rerun-if-changed=runtime/llama.cpp/funasr-capi/funasr_c_api.cpp");
    println!("cargo:rerun-if-changed=runtime/llama.cpp/funasr-common/funasr_sensevoice.h");
    println!("cargo:rerun-if-changed=runtime/llama.cpp/funasr-common/funasr_vad.h");
    println!("cargo:rerun-if-changed=runtime/llama.cpp/funasr-common/funasr_audio.h");

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&runtime_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={build_type}"))
        .arg("-DFUNASR_BUILD_RUST=ON")
        .arg("-DLLAMA_CURL=OFF");
    if let Ok(llama_source) = env::var("FUNASR_LLAMA_SOURCE") {
        configure.arg(format!("-DFETCHCONTENT_SOURCE_DIR_LLAMA={llama_source}"));
    }
    run(configure);

    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg(build_type)
        .arg("--target")
        .arg("funasr_rs")
        .arg("--parallel");
    run(build);

    let lib_dir = build_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=funasr_rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" || target_os == "linux" {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            display_for_linker(&lib_dir)
        );
    }
}

fn display_for_linker(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
