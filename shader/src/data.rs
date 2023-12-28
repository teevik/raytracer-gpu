#[derive(Clone, Copy, Default)]
pub enum Face {
    #[default]
    Front,
    Back,
}

#[derive(Clone, Copy)]
pub struct Range<T: Copy> {
    pub start: T,
    pub end: T,
}

impl<T: Copy + PartialOrd> Range<T> {
    pub fn new(start: T, end: T) -> Self {
        Self { start, end }
    }

    pub fn contains(self, value: T) -> bool {
        value >= self.start && value < self.end
    }
}
