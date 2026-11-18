use lancy::slot_key;
use lancy::support::slotmap::{DenseMap, SlotMap};

slot_key!(Block);

#[test]
fn removing_entities_preserves_keys_and_side_table_alignment() {
    let mut blocks = SlotMap::new();
    let entry: Block = blocks.insert("entry");
    let dead = blocks.insert("dead");
    let exit = blocks.insert("exit");
    assert_eq!(blocks.remove(dead), Some("dead"));
    assert_eq!(blocks.remove(dead), None);

    let extra = blocks.insert("extra");
    assert_ne!(dead, extra);
    assert_eq!(blocks.get(dead), None);
    assert_eq!(blocks.get(Block::INVALID), None);
    assert_eq!(blocks.keys().collect::<Vec<_>>(), [entry, exit, extra]);
    assert_eq!(blocks.len(), 3);

    let mut reachable = DenseMap::filled(blocks.slot_count(), false);
    for key in blocks.keys() {
        reachable[key] = true;
    }
    assert!(reachable[extra]);
    assert!(!reachable[dead]);
    assert_eq!(reachable.get(Block::INVALID), None);
}
