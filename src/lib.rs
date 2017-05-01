// Copyright © 2017 Cormac O'Brien
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#[macro_use]
extern crate approx;
extern crate nalgebra;
extern crate num;
extern crate rand;

pub mod complex;

use complex::Complex64;
use nalgebra::{Matrix, Matrix2, MatrixArray, Vector2};
use num::{One, Zero};
use std::f64;

pub use num::traits::*;
pub use complex::*;

#[derive(Debug)]
pub struct QubedError;

pub fn ket0() -> Vector2<Complex64> {
    let mut ket0 = Matrix::zero();
    ket0[0] = Complex::one();
    ket0
}

pub fn ket1() -> Vector2<Complex64> {
    let mut ket1 = Matrix::zero();
    ket1[1] = Complex::one();
    ket1
}

pub fn qubit0() -> Qubit {
    Qubit { v: ket0() }
}

pub fn qubit1() -> Qubit {
    Qubit { v: ket1() }
}

pub fn hadamard() -> Gate {
    // x = 1 / sqrt(2)
    let x = Complex64::from(0.7071067811865475f64);

    let mut mat: Matrix2<Complex64> = Matrix::zero();

    mat[(0, 0)] = x;
    mat[(0, 1)] = x;
    mat[(1, 0)] = x;
    mat[(1, 1)] = -x;

    Gate { matrix: mat }
}

pub fn pauli_x() -> Gate {
    let mut mat: Matrix2<Complex64> = Matrix::zero();

    mat[(0, 1)] = Complex64::one();
    mat[(1, 0)] = Complex64::one();

    Gate { matrix: mat }
}

pub fn pauli_y() -> Gate {
    let mut mat: Matrix2<Complex64> = Matrix::zero();

    mat[(0, 1)] = -Complex64::i();
    mat[(1, 0)] = Complex64::i();

    Gate { matrix: mat }
}

pub fn pauli_z() -> Gate {
    let mut mat: Matrix2<Complex64> = Matrix::zero();

    mat[(1, 0)] = Complex64::one();
    mat[(0, 1)] = -Complex64::one();

    Gate { matrix: mat }
}

pub fn phase_shift(theta: f64) -> Gate {
    let mut mat: Matrix2<Complex64> = Matrix::zero();

    mat[(1, 0)] = Complex64::one();
    mat[(0, 1)] = (Complex64::i() * theta).exp();

    Gate { matrix: mat }
}

pub fn root_not() -> Gate {
    let mut mat: Matrix2<Complex64> = Matrix::zero();

    let plus = Complex64 { re: 1.0, im: 1.0 };
    let minus = Complex64 {
        re: 1.0,
        im: -1.0,
    };

    mat[(0, 0)] = plus;
    mat[(0, 1)] = minus;
    mat[(1, 0)] = minus;
    mat[(1, 1)] = plus;

    Gate { matrix: mat }
}

pub fn vector_norm(v: &nalgebra::VectorN<Complex64, nalgebra::U2>) -> f64 {
    (v[0].norm().powf(2.0) + v[1].norm().powf(2.0)).sqrt()
}

pub fn c64_relative_eq(x: &Complex64, y: &Complex64) -> bool {
    if x == y {
        return true;
    }

    let norm_diff = (x - y).norm();
    if norm_diff <= f64::EPSILON {
        return true;
    }

    let norm_x = x.norm();
    let norm_y = y.norm();

    let largest = if norm_x > norm_y {
        norm_x
    } else {
        norm_y
    };

    norm_diff <= largest * f64::EPSILON
}

#[derive(Debug)]
pub struct Qubit {
    v: Matrix<Complex64,
              nalgebra::U2,
              nalgebra::U1,
              MatrixArray<Complex64, nalgebra::U2, nalgebra::U1>>,
}


impl Qubit {
    pub fn new(a: Complex64, b: Complex64) -> Result<Qubit, QubedError> {
        let mut v = Matrix::zero();
        v[0] = a;
        v[1] = b;

        if !relative_eq!(vector_norm(&v), 1.0) {
            return Err(QubedError);
        }

        Ok(Qubit { v: v })
    }

    pub fn apply(&self, gate: &Gate) -> Result<Qubit, QubedError> {
        let result = gate.matrix * self.v;
        Qubit::new(result[0], result[1])
    }

    pub fn collapse(&self) -> bool {
        rand::random::<f64>() >= self.v[0].norm().powf(2.0)
    }
}

pub struct Gate {
    pub matrix: Matrix<Complex64,
                       nalgebra::U2,
                       nalgebra::U2,
                       MatrixArray<Complex64, nalgebra::U2, nalgebra::U2>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard() {
        let h = hadamard();

        let q0 = qubit0();
        let result0 = q0.apply(&h).unwrap().apply(&h).unwrap();
        assert!(c64_relative_eq(&result0.v[0], &q0.v[0]));
        assert!(c64_relative_eq(&result0.v[1], &q0.v[1]));

        let q1 = qubit1();
        let result1 = q1.apply(&h).unwrap().apply(&h).unwrap();
        assert!(c64_relative_eq(&result1.v[0], &q1.v[0]));
        assert!(c64_relative_eq(&result1.v[1], &q1.v[1]));
    }

    #[test]
    fn test_pauli_x() {
        let px = pauli_x();

        let q0 = qubit0();
        let result0 = q0.apply(&px).unwrap();
        println!("{:?}", result0);
        assert!(c64_relative_eq(&result0.v[0], &q0.v[1]));
        assert!(c64_relative_eq(&result0.v[1], &q0.v[0]));

        let q1 = qubit1();
        let result1 = q1.apply(&px).unwrap();
        println!("{:?}", result1);
        assert!(c64_relative_eq(&result1.v[0], &q1.v[1]));
        assert!(c64_relative_eq(&result1.v[1], &q1.v[0]));
    }

    #[test]
    fn test_pauli_y() {
        let py = pauli_y();

        let q0 = qubit0();
        let result0 = q0.apply(&py).unwrap();
        println!("{:?}", result0);
        assert!(c64_relative_eq(&result0.v[0], &-(q0.v[1] * Complex64::i())));
        assert!(c64_relative_eq(&result0.v[1], &(q0.v[0] * Complex64::i())));

        let q1 = qubit1();
        let result1 = q1.apply(&py).unwrap();
        println!("{:?}", result1);
        assert!(c64_relative_eq(&result1.v[0], &-(q1.v[1] * Complex64::i())));
        assert!(c64_relative_eq(&result1.v[1], &(q1.v[0] * Complex64::i())));
    }

    #[test]
    fn test_pauli_z() {
        let py = pauli_z();

        let q0 = qubit0();
        let result0 = q0.apply(&py).unwrap();
        println!("{:?}", result0);
        assert!(c64_relative_eq(&result0.v[0], &-q0.v[1]));
        assert!(c64_relative_eq(&result0.v[1], &q0.v[0]));

        let q1 = qubit1();
        let result1 = q1.apply(&py).unwrap();
        println!("{:?}", result1);
        assert!(c64_relative_eq(&result1.v[0], &-q1.v[1]));
        assert!(c64_relative_eq(&result1.v[1], &q1.v[0]));
    }
}
