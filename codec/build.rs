//! Builds the AVX2 kernels (simd/reel_simd.c) with clang or cc, and with the
//! `cuda` feature the motion-search kernel (cuda/me.cu) with nvcc.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(reel_simd)");
    println!("cargo:rerun-if-changed=../simd/reel_simd.c");
    println!("cargo:rerun-if-changed=../cuda/me.cu");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let windows = std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows";
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    if std::env::var("CARGO_FEATURE_SIMD").is_ok() && arch == "x86_64" {
        build_simd(&out, windows);
    }
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda(&out, windows);
    }
}

fn build_simd(out: &PathBuf, windows: bool) {
    let obj = out.join("reel_simd.o");
    let llvm = |tool: &str| {
        let p = format!("C:/Program Files/LLVM/bin/{tool}.exe");
        if std::path::Path::new(&p).exists() { p } else { tool.to_string() }
    };
    let (cc, ar, lib) = if windows {
        (llvm("clang"), llvm("llvm-ar"), out.join("reelsimd.lib"))
    } else {
        (std::env::var("CC").unwrap_or_else(|_| "cc".into()), "ar".to_string(), out.join("libreelsimd.a"))
    };
    let mut c = Command::new(&cc);
    c.args(["-O2", "-mavx2", "-c", "../simd/reel_simd.c", "-o"]).arg(&obj);
    if windows {
        c.arg("--target=x86_64-pc-windows-msvc");
    } else {
        c.arg("-fPIC");
    }
    let ok = c.status().map(|s| s.success()).unwrap_or(false)
        && Command::new(&ar).arg("rcs").arg(&lib).arg(&obj).status().map(|s| s.success()).unwrap_or(false);
    if !ok {
        println!("cargo:warning=no C compiler for the AVX2 kernels: using the Rust versions only");
        return;
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=reelsimd");
    println!("cargo:rustc-cfg=reel_simd");
}

fn build_cuda(out: &PathBuf, windows: bool) {
    let lib = out.join(if windows { "reelcuda.lib" } else { "libreelcuda.a" });
    let arch = std::env::var("REEL_CUDA_ARCH").unwrap_or_else(|_| "sm_75".into());
    let mut cmd = Command::new("nvcc");
    cmd.args(["-O3", "-std=c++17", "--lib", &format!("-arch={arch}"), "-o"]).arg(&lib).arg("../cuda/me.cu");
    if windows {
        cmd.args(["-Xcompiler", "/MD,/Zc:preprocessor"]);
        // nvcc wants cl.exe; outside a developer prompt, ask vswhere where it is
        if let Some(cl) = std::env::var("REEL_CCBIN").ok().or_else(find_cl) {
            cmd.args(["-ccbin", &cl]);
        }
    } else {
        cmd.args(["-Xcompiler", "-fPIC"]);
    }
    let st = cmd.status().expect("nvcc not found (the cuda feature needs the CUDA toolkit)");
    assert!(st.success(), "nvcc failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=reelcuda");
    let cuda = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".into());
    let libdir = PathBuf::from(&cuda).join(if windows { "lib/x64" } else { "lib64" });
    println!("cargo:rustc-link-search=native={}", libdir.display());
    println!("cargo:rustc-link-lib=static=cudart_static");
    if !windows {
        for l in ["dl", "rt", "pthread", "stdc++"] {
            println!("cargo:rustc-link-lib={l}");
        }
    }
}

fn find_cl() -> Option<String> {
    let vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe";
    let out = Command::new(vswhere)
        .args(["-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])
        .output()
        .ok()?;
    let root = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()).join(r"VC\Tools\MSVC");
    let mut versions: Vec<_> = std::fs::read_dir(&root).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    versions.sort();
    let bin = versions.last()?.join(r"bin\Hostx64\x64");
    bin.join("cl.exe").exists().then(|| bin.display().to_string())
}
