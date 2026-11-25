use crate::{Function, VariableLocation};
use gimli::write::{
    Address, AttributeValue as A, DwarfUnit, EndianVec, Expression, FrameTable, LineProgram,
    LineString, Range, RangeList, Sections, Unit, UnitEntryId,
};
use gimli::{self as dw, Encoding, Format, LineEncoding, LittleEndian, Register};
use splice_core::debug::{ConstantType, Location, Type};
use splice_core::hir::ScalarType;
use splice_core::simd::Element;
use std::path::{Path, PathBuf};

pub(crate) const ENCODING: Encoding = Encoding {
    format: Format::Dwarf32,
    version: 4,
    address_size: 8,
};

fn source_path(source: Location) -> String {
    let file = Path::new(source.file);
    if file.is_absolute() {
        return source.file.replace('\\', "/");
    }
    let manifest = Path::new(source.directory);
    // rustc can make file! relative to the workspace or to the package.
    manifest
        .ancestors()
        .map(|dir| dir.join(file))
        .find(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from(source.directory).join(file))
        .to_string_lossy()
        .replace('\\', "/")
}

fn name(unit: &mut Unit, id: UnitEntryId, value: &str) {
    unit.get_mut(id)
        .set(dw::DW_AT_name, A::String(value.as_bytes().to_vec()));
}

fn address_range(unit: &mut Unit, id: UnitEntryId, begin: u64, end: u64) {
    let die = unit.get_mut(id);
    die.set(dw::DW_AT_low_pc, A::Address(Address::Constant(begin)));
    die.set(dw::DW_AT_high_pc, A::Udata(end - begin));
}

fn base(unit: &mut Unit, label: &str, bytes: usize, encoding: dw::DwAte) -> UnitEntryId {
    let id = unit.add(unit.root(), dw::DW_TAG_base_type);
    name(unit, id, label);
    unit.get_mut(id)
        .set(dw::DW_AT_encoding, A::Encoding(encoding));
    unit.get_mut(id)
        .set(dw::DW_AT_byte_size, A::Udata(bytes as u64));
    id
}

fn element(unit: &mut Unit, elem: Element) -> UnitEntryId {
    let encoding = match elem {
        Element::F32 | Element::F64 => dw::DW_ATE_float,
        Element::I32 | Element::I64 => dw::DW_ATE_signed,
        Element::U16 | Element::U32 => dw::DW_ATE_unsigned,
    };
    base(unit, elem.name(), elem.bytes(), encoding)
}

fn emit_type(unit: &mut Unit, ty: Type) -> UnitEntryId {
    match ty {
        Type::Scalar(ty) => {
            let encoding = match ty {
                ScalarType::Bool => dw::DW_ATE_boolean,
                ScalarType::F32 | ScalarType::F64 => dw::DW_ATE_float,
                ScalarType::I32 | ScalarType::I64 => dw::DW_ATE_signed,
                ScalarType::U32 | ScalarType::U64 | ScalarType::Index => dw::DW_ATE_unsigned,
            };
            base(unit, ty.name(), ty.bytes(), encoding)
        }
        Type::Constant(kind) => match kind {
            ConstantType::Bool => base(unit, "bool", 1, dw::DW_ATE_boolean),
            ConstantType::Usize => base(unit, "usize", 8, dw::DW_ATE_unsigned),
            ConstantType::Signed(bytes) => base(
                unit,
                &format!("i{}", bytes * 8),
                usize::from(bytes),
                dw::DW_ATE_signed,
            ),
            ConstantType::Unsigned(bytes) => base(
                unit,
                &format!("u{}", bytes * 8),
                usize::from(bytes),
                dw::DW_ATE_unsigned,
            ),
            ConstantType::Float(bytes) => base(
                unit,
                &format!("f{}", bytes * 8),
                usize::from(bytes),
                dw::DW_ATE_float,
            ),
        },
        Type::Vector {
            element: elem,
            lanes,
        } => {
            let elem_ty = element(unit, elem);
            let id = unit.add(unit.root(), dw::DW_TAG_array_type);
            unit.get_mut(id).set(dw::DW_AT_type, A::UnitRef(elem_ty));
            let count = unit.add(id, dw::DW_TAG_subrange_type);
            unit.get_mut(count)
                .set(dw::DW_AT_count, A::Udata(lanes as u64));
            id
        }
        Type::Slice(elem) => {
            let elem_ty = element(unit, elem);
            let ptr = unit.add(unit.root(), dw::DW_TAG_pointer_type);
            unit.get_mut(ptr).set(dw::DW_AT_type, A::UnitRef(elem_ty));
            unit.get_mut(ptr).set(dw::DW_AT_byte_size, A::Udata(8));
            let len = base(unit, "usize", 8, dw::DW_ATE_unsigned);
            let id = unit.add(unit.root(), dw::DW_TAG_structure_type);
            name(unit, id, &format!("&[{}]", elem.name()));
            unit.get_mut(id).set(dw::DW_AT_byte_size, A::Udata(16));
            for (label, ty, offset) in [("data_ptr", ptr, 0), ("length", len, 8)] {
                let field = unit.add(id, dw::DW_TAG_member);
                name(unit, field, label);
                unit.get_mut(field).set(dw::DW_AT_type, A::UnitRef(ty));
                unit.get_mut(field)
                    .set(dw::DW_AT_data_member_location, A::Udata(offset));
            }
            id
        }
    }
}

pub(super) fn sections(
    functions: &[Function],
    frames: &FrameTable,
) -> gimli::write::Result<Sections<EndianVec<LittleEndian>>> {
    let mut dwarf = DwarfUnit::new(ENCODING);
    let unit = &mut dwarf.unit;
    let root = unit.root();
    let first = functions.iter().flat_map(|f| &f.statements).next();
    let file = first.map_or_else(|| "splice".into(), |stmt| source_path(stmt.source));
    let dir = first.map_or(".", |stmt| stmt.source.directory);
    name(unit, root, &file);
    unit.get_mut(root)
        .set(dw::DW_AT_comp_dir, A::String(dir.as_bytes().to_vec()));
    unit.get_mut(root)
        .set(dw::DW_AT_producer, A::String(b"Splice".to_vec()));
    unit.get_mut(root)
        .set(dw::DW_AT_language, A::Language(dw::DW_LANG_Rust));
    let ranges = RangeList(
        functions
            .iter()
            .map(|f| Range::StartLength {
                begin: Address::Constant(f.address),
                length: f.size,
            })
            .collect(),
    );
    let ranges = unit.ranges.add(ranges);
    unit.get_mut(root)
        .set(dw::DW_AT_ranges, A::RangeListRef(ranges));
    unit.line_program = LineProgram::new(
        ENCODING,
        LineEncoding::default(),
        LineString::String(dir.as_bytes().to_vec()),
        None,
        LineString::String(file.into_bytes()),
        None,
    );
    let mut types = std::collections::HashMap::new();
    let mut files = std::collections::HashMap::new();
    for function in functions {
        let id = unit.add(root, dw::DW_TAG_subprogram);
        name(unit, id, &function.name);
        address_range(unit, id, function.address, function.address + function.size);
        unit.get_mut(id).set(dw::DW_AT_external, A::Flag(true));
        if function.statements.is_empty() {
            continue;
        }
        let mut statements: Vec<_> = function.statements.iter().collect();
        statements.sort_by_key(|stmt| stmt.address);
        let first = statements[0];
        let source = first.source;
        let mut file_id = |unit: &mut Unit, location: Location| {
            *files
                .entry((location.directory, location.file))
                .or_insert_with(|| {
                    let dir = unit.line_program.default_directory();
                    unit.line_program.add_file(
                        LineString::String(source_path(location).into_bytes()),
                        dir,
                        None,
                    )
                })
        };
        let file = file_id(unit, source);
        unit.get_mut(id)
            .set(dw::DW_AT_decl_file, A::FileIndex(Some(file)));
        unit.get_mut(id)
            .set(dw::DW_AT_decl_line, A::Udata(u64::from(source.line)));
        unit.line_program
            .begin_sequence(Some(Address::Constant(function.address)));
        if first.address != function.address {
            let row = unit.line_program.row();
            row.file = file;
            row.line = u64::from(source.line);
            row.column = u64::from(source.column);
            row.is_statement = false;
            unit.line_program.generate_row();
        }
        let mut statements = statements.into_iter().peekable();
        while let Some(stmt) = statements.next() {
            let end = statements
                .peek()
                .map_or(function.address + function.size, |next| next.address);
            if end == stmt.address {
                continue;
            }
            let location = stmt.source;
            let file = file_id(unit, location);
            let row = unit.line_program.row();
            row.address_offset = stmt.address - function.address;
            row.file = file;
            row.line = u64::from(location.line);
            row.column = u64::from(location.column);
            row.is_statement = true;
            row.prologue_end = stmt.address == first.address;
            unit.line_program.generate_row();
            let scope = unit.add(id, dw::DW_TAG_lexical_block);
            address_range(unit, scope, stmt.address, end);
            for &(label, kind, ref location) in &stmt.variables {
                let ty = *types.entry(kind).or_insert_with(|| emit_type(unit, kind));
                let var = unit.add(scope, dw::DW_TAG_variable);
                name(unit, var, label);
                let die = unit.get_mut(var);
                die.set(dw::DW_AT_type, A::UnitRef(ty));
                match *location {
                    VariableLocation::Memory(offset) => {
                        let mut expression = Expression::new();
                        expression.op_breg(Register(stmt.register), offset);
                        die.set(dw::DW_AT_location, A::Exprloc(expression));
                    }
                    VariableLocation::Constant(bits) => {
                        let bytes = match kind {
                            Type::Constant(ConstantType::Bool) => 1,
                            Type::Constant(ConstantType::Usize) => 8,
                            Type::Constant(
                                ConstantType::Signed(n)
                                | ConstantType::Unsigned(n)
                                | ConstantType::Float(n),
                            ) => usize::from(n),
                            _ => unreachable!("constant location requires constant type"),
                        };
                        die.set(
                            dw::DW_AT_const_value,
                            A::Block(bits.to_le_bytes()[..bytes].to_vec()),
                        );
                    }
                }
            }
        }
        unit.line_program.end_sequence(function.size);
    }
    let mut sections = Sections::new(EndianVec::new(LittleEndian));
    dwarf.write(&mut sections)?;
    frames.write_debug_frame(&mut sections.debug_frame)?;
    Ok(sections)
}
