# Qubed

Qubed is a simple quantum computing library under active development. Currently
it simulates single `Qubit`s and `Gate`s which act on them.

This project makes use of a slightly modified version of the `num` crate
(specifically `num::complex`), adding implementations of `AddAssign` et. al. in
order to allow `nalgebra` to use `Complex` types. The license for that code can
be found in `complex.rs`.
