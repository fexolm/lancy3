//! Read the symbol file back with independent ELF and DWARF readers.
use crate::{Function, Statement, VariableLocation, dwarf, elf};
use gimli::write::FrameTable;
use object::{Object, ObjectSection, ObjectSymbol};
use splice_core::debug::{Location, Type};
use splice_core::hir::ScalarType;

fn source(line: u32) -> Location {
    Location {
        file: "src/kernel.rs",
        directory: "/work",
        line,
        column: 5,
    }
}

fn functions() -> Vec<Function> {
    let statement = |address, line, variables| Statement {
        address,
        source: source(line),
        register: 7,
        variables,
    };
    vec![
        Function {
            name: "root".into(),
            address: 0x1000,
            size: 0x40,
            statements: vec![
                statement(
                    0x1000,
                    10,
                    vec![(
                        "x",
                        Type::Scalar(ScalarType::I32),
                        VariableLocation::Memory(-8),
                    )],
                ),
                statement(
                    0x1010,
                    11,
                    vec![(
                        "flag",
                        Type::Scalar(ScalarType::Bool),
                        VariableLocation::Memory(16),
                    )],
                ),
            ],
        },
        Function {
            name: "child".into(),
            address: 0x2000,
            size: 0x20,
            statements: vec![statement(0x2008, 20, Vec::new())],
        },
    ]
}

#[test]
fn symbol_file_parses_as_elf_with_symbols_lines_and_variables() {
    let functions = functions();
    let sections = dwarf::sections(&functions, &FrameTable::default()).unwrap();
    let bytes = elf::symbols(&functions, &sections).unwrap();
    let file = object::File::parse(&*bytes).unwrap();

    let mut symbols: Vec<_> = file
        .symbols()
        .filter(|s| s.kind() == object::SymbolKind::Text)
        .map(|s| (s.name().unwrap().to_owned(), s.address(), s.size()))
        .collect();
    symbols.sort();
    assert_eq!(
        symbols,
        [
            ("child".into(), 0x2000, 0x20),
            ("root".into(), 0x1000, 0x40)
        ]
    );

    let load = |id: gimli::SectionId| -> Result<_, gimli::Error> {
        let data = file
            .section_by_name(id.name())
            .map_or(&[][..], |s| s.data().unwrap());
        Ok(gimli::EndianSlice::new(data, gimli::LittleEndian))
    };
    let dwarf = gimli::Dwarf::load(load).unwrap();
    let unit = dwarf.units().next().unwrap().unwrap();
    let unit = dwarf.unit(unit).unwrap();

    let mut rows = Vec::new();
    let program = unit.line_program.clone().unwrap();
    let mut program = program.rows();
    while let Some((_, row)) = program.next_row().unwrap() {
        if row.is_stmt() && !row.end_sequence() {
            rows.push((row.address(), row.line().unwrap().get()));
        }
    }
    assert_eq!(rows, [(0x1000, 10), (0x1010, 11), (0x2008, 20)]);

    let mut names = Vec::new();
    let mut entries = unit.entries();
    while let Some(entry) = entries.next_dfs().unwrap() {
        let tag = entry.tag();
        if tag == gimli::DW_TAG_subprogram || tag == gimli::DW_TAG_variable {
            let name = entry.attr_value(gimli::DW_AT_name).unwrap();
            let name = dwarf.attr_string(&unit, name).unwrap();
            let location = entry.attr(gimli::DW_AT_location).is_some();
            names.push((tag, name.to_string().unwrap().to_owned(), location));
        }
    }
    use gimli::{DW_TAG_subprogram as F, DW_TAG_variable as V};
    assert_eq!(
        names,
        [
            (F, "root".into(), false),
            (V, "x".into(), true),
            (V, "flag".into(), true),
            (F, "child".into(), false),
        ]
    );
}
