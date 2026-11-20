use std::{collections::BTreeMap, env, fs, path::Path};

// libtest registers functions at compile time. Track the directory as well as
// its contents so adding or renaming a .lancy file updates cargo test --list.
fn discover(root: &Path, dir: &Path, cases: &mut BTreeMap<String, Vec<String>>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir()
            && (path.file_name().unwrap() != "aarch64"
                || env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64"))
        {
            discover(root, &path, cases);
        } else if path.extension().is_some_and(|ext| ext == "lancy") {
            let relative = path
                .strip_prefix(root)
                .unwrap()
                .to_str()
                .unwrap()
                .replace('\\', "/");
            let stem = relative.strip_suffix(".lancy").unwrap();
            let words: Vec<_> = stem
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|part| !part.is_empty())
                .collect();
            let name = format!("ir_{}", words.join("_").to_ascii_lowercase());
            cases.entry(name).or_default().push(relative);
        }
    }
}

fn main() {
    let root = Path::new("filetests");
    println!("cargo::rerun-if-changed=filetests");
    let mut cases = BTreeMap::new();
    discover(root, root, &mut cases);
    assert!(!cases.is_empty(), "no .lancy tests found");
    let mut tests = String::new();
    for (name, mut files) in cases {
        files.sort();
        let collision = files.len() > 1;
        if collision {
            tests.push_str(&format!("mod {name} {{ use super::fixture;\n"));
        }
        for (i, file) in files.iter().enumerate() {
            if file.starts_with("passes/experimental/") {
                tests.push_str("#[cfg(feature = \"experimental-simd\")]\n");
            }
            let test = if collision {
                format!("case_{}", i + 1)
            } else {
                name.clone()
            };
            tests.push_str(&format!(
                "#[test]\nfn {test}() {{ fixture({file:?}, include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/filetests/\", {file:?}))); }}\n"
            ));
        }
        if collision {
            tests.push_str("}\n");
        }
    }
    fs::write(
        Path::new(&env::var_os("OUT_DIR").unwrap()).join("filetests.rs"),
        tests,
    )
    .unwrap();
}
