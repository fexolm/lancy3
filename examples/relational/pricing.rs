use super::{Combine, Expr, Plan, Predicate};

// Five relations keyed by product_id. Compute quantity * price on the left,
// promotional adjustments on the right, then combine both per matching row.
pub fn pricing_plan() -> Plan {
    use Expr::*;
    use Plan::*;
    let sales = Filter(Box::new(Scan(0)), Predicate::AtLeast(Value, Literal(2)));
    let revenue = Join(Box::new(sales), Box::new(Scan(1)), Combine::Multiply);
    let eligible = Filter(
        Box::new(Scan(2)),
        Predicate::And(
            Box::new(Predicate::Equal(Value, Literal(1))),
            Box::new(Predicate::AtLeast(Key, Literal(0))),
        ),
    );
    let promotions = Join(Box::new(eligible), Box::new(Scan(3)), Combine::Add);
    let adjustments = Project(
        Box::new(Scan(4)),
        Key,
        Min(
            Box::new(Mul(Box::new(Value), Box::new(Literal(2)))),
            Box::new(Literal(20)),
        ),
    );
    let discounts = Join(
        Box::new(promotions),
        Box::new(adjustments),
        Combine::Multiply,
    );
    Project(
        Box::new(Join(Box::new(revenue), Box::new(discounts), Combine::Add)),
        Key,
        Add(Box::new(Value), Box::new(Literal(1))),
    )
}
