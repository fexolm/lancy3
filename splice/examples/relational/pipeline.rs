use super::{Combine, Expr, Plan, Predicate};

pub(super) enum Step<'a> {
    Filter(&'a Predicate),
    Project(&'a Expr, &'a Expr),
    Find(usize),
    Values(usize, Combine),
}

#[derive(splice::KernelType)]
pub(super) struct Pipe<'a> {
    pub table: usize,
    pub steps: Box<[Step<'a>]>,
    pub index: Option<usize>,
    pub high: bool,
}

impl Pipe<'_> {
    fn state_words(&self, lanes: usize) -> usize {
        // Each boundary and each retry queue has a count and three packets.
        (2 * self.steps.len() + 1) * (1 + 3 * lanes)
    }
}

pub(super) enum Task<'a> {
    Save(usize, bool),
    Run(Pipe<'a>),
}

#[derive(splice::KernelType)]
pub struct Program<'a> {
    pub(super) tasks: Vec<Task<'a>>,
    pub(super) indices: usize,
    pub(super) lanes: usize,
    pub(super) state_words: usize,
}

impl Plan {
    pub fn prepare(&self) -> Program<'_> {
        Program::new(self)
    }

    pub fn state_words(&self) -> usize {
        self.prepare().state_words
    }
}

impl<'a> Program<'a> {
    fn required_words(&self) -> usize {
        let lanes = self.lanes;
        let largest_pipe = self
            .tasks
            .iter()
            .filter_map(|task| match task {
                Task::Run(pipe) => Some(pipe.state_words(lanes)),
                Task::Save(..) => None,
            })
            .max()
            .unwrap_or(0);
        4 * self.indices + largest_pipe
    }

    pub fn new(plan: &'a Plan) -> Self {
        let mut program = Self {
            tasks: Vec::new(),
            indices: 0,
            lanes: splice::simd::lanes::<i32>(),
            state_words: 0,
        };
        program.pipe(plan, None, false);
        program.state_words = program.required_words();
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
            steps: steps.into_boxed_slice(),
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
