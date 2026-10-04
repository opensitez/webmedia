//! Opt-in, per-thread test instrumentation; absent from production builds.
use std::cell::Cell;
use std::time::Instant;

std::thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static TIMES: Cell<[u128; 5]> = const { Cell::new([0; 5]) };
    static STAGES: Cell<[u128; 4]> = const { Cell::new([0; 4]) };
}

pub(crate) struct Measure(usize, Option<Instant>);
impl Drop for Measure {
    fn drop(&mut self) {
        if let Some(start) = self.1 {
            TIMES.with(|times| {
                let mut values = times.get();
                values[self.0] += start.elapsed().as_nanos();
                times.set(values);
            });
        }
    }
}
pub(crate) fn measure(category: usize) -> Measure {
    Measure(
        category,
        ENABLED.with(|enabled| enabled.get().then(Instant::now)),
    )
}
pub(crate) fn reset(enabled: bool) {
    ENABLED.with(|flag| flag.set(enabled));
    TIMES.with(|times| times.set([0; 5]));
    STAGES.with(|times| times.set([0; 4]));
}
pub(crate) fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
pub(crate) fn record_stage(index: usize, nanos: u128) {
    STAGES.with(|times| {
        let mut values = times.get();
        values[index] = nanos;
        times.set(values);
    });
}
pub(crate) fn stages() -> [u128; 4] {
    STAGES.with(Cell::get)
}
pub(crate) fn times() -> [u128; 5] {
    TIMES.with(Cell::get)
}
