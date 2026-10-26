//! Criterion benchmarks for the pure Rust `chronon_divmod` kernels.
//!
//! Inputs are generated with the same deterministic LCG used by the R
//! benchmark script (`benchmarks/bench.R`):
//!
//!   v = ((i * 1103515245 + 12345) mod 2^31) mod modulus - offset
//!
//! so both sides operate on identical vectors without shipping large files.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mixtime_rs::{day_to_month, day_to_year, month_to_day, year_to_day};
use numpy::ndarray::ArrayView1;

const SIZES: [usize; 3] = [1_000, 100_000, 1_000_000];

fn lcg_vec(n: usize, modulus: u64, offset: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let v = ((i as u64).wrapping_mul(1_103_515_245).wrapping_add(12_345) % (1u64 << 31))
                % modulus;
            v as f64 - offset
        })
        .collect()
}

fn bench_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixtime_kernel");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for &n in &SIZES {
        let days = lcg_vec(n, 200_001, 100_000.0);
        let months = lcg_vec(n, 10_001, 5_000.0);
        let years = lcg_vec(n, 1_001, 500.0);

        // More samples for small inputs, fewer for the 1M case.
        group.sample_size(if n >= 1_000_000 { 20 } else { 50 });
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("day_to_month", n), &days, |b, x| {
            b.iter(|| day_to_month(black_box(ArrayView1::from(&x[..])), 1.0, 1.0))
        });
        group.bench_with_input(BenchmarkId::new("day_to_year", n), &days, |b, x| {
            b.iter(|| day_to_year(black_box(ArrayView1::from(&x[..])), 1.0, 1.0))
        });
        group.bench_with_input(BenchmarkId::new("month_to_day", n), &months, |b, x| {
            b.iter(|| month_to_day(black_box(ArrayView1::from(&x[..])), 1.0, 1.0))
        });
        group.bench_with_input(BenchmarkId::new("year_to_day", n), &years, |b, x| {
            b.iter(|| year_to_day(black_box(ArrayView1::from(&x[..])), 1.0, 1.0))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_kernel);
criterion_main!(benches);
