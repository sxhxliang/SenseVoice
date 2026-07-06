use std::collections::hash_map::DefaultHasher;
use std::env;
use std::hash::{Hash, Hasher};
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
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let runtime_dir = manifest_dir.join("runtime").join("llama.cpp");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
    let build_dir = out_dir.join(format!("funasr-cmake-{:x}", path_hash(&runtime_dir)));
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
    println!("cargo:rerun-if-changed=runtime/llama.cpp/funasr-common/miniaudio.h");
    println!("cargo:rerun-if-changed=runtime/llama.cpp/funasr-sensevoice/funasr-sensevoice.cpp");
    println!("cargo:rerun-if-changed=runtime/llama.cpp/funasr-vad/funasr-vad.cpp");

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&runtime_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={build_type}"))
        .arg("-DFUNASR_BUILD_RUST=ON")
        .arg("-DLLAMA_CURL=OFF");
    add_llama_source_override(&mut configure);
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

    let lib_dirs = native_library_dirs(&build_dir, build_type);
    for lib_dir in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=funasr_rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" || target_os == "linux" {
        if let Some(lib_dir) = lib_dirs.first() {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                display_for_linker(lib_dir)
            );
        }
    }
}

fn add_llama_source_override(configure: &mut Command) {
    let Some(llama_source) = env::var_os("FUNASR_LLAMA_SOURCE") else {
        return;
    };
    let llama_source = PathBuf::from(llama_source);
    if !llama_source.join("CMakeLists.txt").is_file() {
        panic!(
            "FUNASR_LLAMA_SOURCE must point to a llama.cpp checkout containing CMakeLists.txt: {}",
            llama_source.display()
        );
    }
    configure.arg(format!(
        "-DFETCHCONTENT_SOURCE_DIR_LLAMA={}",
        llama_source.display()
    ));
}

fn native_library_dirs(build_dir: &Path, build_type: &str) -> Vec<PathBuf> {
    let lib_dir = build_dir.join("lib");
    let config_lib_dir = lib_dir.join(build_type);

    let mut dirs = Vec::new();
    if config_lib_dir.is_dir() {
        dirs.push(config_lib_dir);
    }
    dirs.push(lib_dir);
    dirs
}

fn display_for_linker(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn path_hash(path: &Path) -> u64 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}
