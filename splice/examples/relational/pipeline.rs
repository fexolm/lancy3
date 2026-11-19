use super::{Combine, Expr, Plan, Predicate};

pub(super) enum Step<'a> {
    Filter(&'a Predicate),
    Project(&'a Expr, &'a Expr),
    Find(usize),
    Values(usize, Combine),
}

impl Step<'_> {
    pub fn carries_heads(&self) -> bool {
        matches!(self, Self::Values(..))
    }
}

pub(super) struct Pipe<'a> {
    pub table: usize,
    pub steps: Vec<Step<'a>>,
    pub index: Option<usize>,
    pub high: bool,
}

impl Pipe<'_> {
    pub fn dependencies(&self) -> impl Iterator<Item = usize> + '_ {
        self.steps.iter().filter_map(|step| match step {
            Step::Find(id) => Some(*id),
            _ => None,
        })
    }
}

pub(super) enum Task<'a> {
    Save(usize, bool),
    Run(Pipe<'a>),
}

pub(super) struct Program<'a> {
    pub tasks: Vec<Task<'a>>,
    pub indices: usize,
}

impl<'a> Program<'a> {
    pub fn new(plan: &'a Plan) -> Self {
        let mut program = Self {
            tasks: Vec::new(),
            indices: 0,
        };
        program.pipe(plan, None, false);
        program
    }

    fn pipe(&mut self, plan: &'a Plan, index: Option<usize>, high: bool) {
        if let Some(id) = index {
            self.tasks.push(Task::Save(id, high));
        }
        let mut steps = Vec::new();
        let table = self.walk(plan, &mut steps, high);
        self.tasks.push(Task::Run(Pipe {
            table,
            steps,
            index,
            high,
        }));
    }

    fn walk(&mut self, plan: &'a Plan, steps: &mut Vec<Step<'a>>, high: bool) -> usize {
        match plan {
            Plan::Scan(table) => *table,
            Plan::Filter(input, pred) => {
                let table = self.walk(input, steps, high);
                steps.push(Step::Filter(pred));
                table
            }
            Plan::Project(input, k, v) => {
                let table = self.walk(input, steps, high);
                steps.push(Step::Project(k, v));
                table
            }
            Plan::Join(left, right, combine) => {
                let table = self.walk(left, steps, high);
                let id = self.indices;
                self.indices += 1;
                self.pipe(right, Some(id), !high);
                steps.push(Step::Find(id));
                steps.push(Step::Values(id, *combine));
                table
            }
        }
    }
}
