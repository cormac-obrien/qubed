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
use nalgebra::{DMatrix, DVector, Matrix, MatrixArray, Vector2};
use num::{One, Zero};
use std::error::Error;
use std::{f64, fmt};
use std::ops::Index;

pub use num::traits::*;
pub use complex::*;

const MAX_WIDTH: usize = 8;

#[derive(Debug)]
pub enum QubedError {
    InvalidNorm(f64),
    UnsupportedWidth(usize),
    IncompatibleDimensions(usize, usize),
}

impl fmt::Display for QubedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            QubedError::IncompatibleDimensions(n, m) => write!(f, "Attempted to multiply matrix with {} columns by matrix with {} rows", n, m),
            QubedError::InvalidNorm(n) => write!(f, "Norm of register not equal to 1 (was {})", n),
            QubedError::UnsupportedWidth(w) => write!(f, "Cannot create a register of width {}", w),
        }
    }
}

impl Error for QubedError {
    fn description(&self) -> &str {
        match *self {
            QubedError::IncompatibleDimensions(_, _) => "Attempted to multiply matrices with incompatible dimensions",
            QubedError::InvalidNorm(_) => "Norm of register not equal to 1",
            QubedError::UnsupportedWidth(_) => "Cannot create a register of the given width",
        }
    }

    fn cause(&self) -> Option<&Error> {
        match *self {
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Register {
    vector: DVector<Complex64>,
    width: usize,
}

impl Register {
    /// Creates a new quantum register initialized to zero
    pub fn new(width: usize) -> Result<Register, QubedError> {
        if width == 0 || width > MAX_WIDTH {
            return Err(QubedError::UnsupportedWidth(width));
        }

        let mut vector = DVector::from_element(2.pow(width as u32), Complex::zero());

        vector[0] = Complex64::one();

        Ok(Register { vector, width })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn apply(&self, gate: &Gate) -> Result<Register, QubedError> {
        if gate.matrix.ncols() != self.vector.nrows() {
            return Err(QubedError::IncompatibleDimensions(gate.matrix.ncols(), self.vector.nrows()));
        }

        let vector = &gate.matrix * &self.vector;

        let norm = vector_norm(&vector);
        if !relative_eq!(norm, 1.0, epsilon = f64::EPSILON) {
            return Err(QubedError::InvalidNorm(norm));
        }

        Ok(Register { vector: vector, width: self.width })
    }
}

impl Index<usize> for Register {
    type Output = Complex64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vector[index]
    }
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
        let mut v = Vector2::zero();
        v[0] = a;
        v[1] = b;

        let norm = (a.norm().powf(2.0) + b.norm().powf(2.0)).sqrt();
        if !relative_eq!(norm, 1.0) {
            return Err(QubedError::InvalidNorm(norm));
        }

        Ok(Qubit { v: v })
    }

    pub fn apply(&self, gate: &Gate) -> Result<Qubit, QubedError> {
        let result = &gate.matrix * self.v;
        Qubit::new(result[0], result[1])
    }

    pub fn collapse(&self) -> bool {
        rand::random::<f64>() >= self.v[0].norm().powf(2.0)
    }
}

pub struct Gate {
    pub matrix: DMatrix<Complex64>,
}

pub fn kronecker_product(a: &DMatrix<Complex64>, b: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    let mut result = DMatrix::from_element(a.nrows() * b.nrows(), a.ncols() * b.ncols(), Complex64::zero());

    let brows = b.nrows();
    let bcols = b.ncols();

    for ar in 0..a.nrows() {
        for ac in 0..a.ncols() {
            for br in 0..b.nrows() {
                for bc in 0..b.ncols() {
                    result[(ar * brows + br, ac * bcols + bc)] = a[(ar, ac)] * b[(br, bc)];
                }
            }
        }
    }

    for r in 0..result.nrows() {
        for c in 0..result.ncols() {
            print!("{:+.2} ", result[(r, c)]);
        }
        println!("");
    }

    result
}

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

pub fn hadamard(qubits: usize) -> Gate {
    // x = 1 / sqrt(2)
    let x = Complex64::from(0.7071067811865475f64);

    let mut mat = DMatrix::from_element(2, 2, Complex64::zero());

    mat[(0, 0)] = x;
    mat[(0, 1)] = x;
    mat[(1, 0)] = x;
    mat[(1, 1)] = -x;

    let copy = mat.clone();

    for _ in 1..qubits {
        mat = kronecker_product(&mat, &copy);
    }

    Gate { matrix: mat }
}

pub fn pauli_x() -> Gate {
    let mut mat = DMatrix::from_element(2, 2, Complex64::zero());

    mat[(0, 1)] = Complex64::one();
    mat[(1, 0)] = Complex64::one();

    Gate { matrix: mat }
}

pub fn pauli_y() -> Gate {
    let mut mat = DMatrix::from_element(2, 2, Complex64::zero());

    mat[(0, 1)] = -Complex64::i();
    mat[(1, 0)] = Complex64::i();

    Gate { matrix: mat }
}

pub fn pauli_z() -> Gate {
    let mut mat = DMatrix::from_element(2, 2, Complex64::zero());

    mat[(1, 0)] = Complex64::one();
    mat[(0, 1)] = -Complex64::one();

    Gate { matrix: mat }
}

pub fn phase_shift(theta: f64) -> Gate {
    let mut mat = DMatrix::from_element(2, 2, Complex64::zero());

    mat[(1, 0)] = Complex64::one();
    mat[(0, 1)] = (Complex64::i() * theta).exp();

    Gate { matrix: mat }
}

pub fn root_not() -> Gate {
    let mut mat = DMatrix::from_element(2, 2, Complex64::zero());

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

pub fn vector_norm(v: &DVector<Complex64>) -> f64 {
    let mut sum = 0.0;

    for i in 0..v.len() {
        sum += v[i].norm().powf(2.0);
    }

    sum.sqrt()
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

    if norm_diff <= largest * f64::EPSILON {
        true
    } else {
        println!("{} != {}", x, y);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard() {
        let h1 = hadamard(1);

        let q0 = qubit0();
        let result0 = q0.apply(&h1).unwrap().apply(&h1).unwrap();
        assert!(c64_relative_eq(&result0.v[0], &q0.v[0]));
        assert!(c64_relative_eq(&result0.v[1], &q0.v[1]));

        let q1 = qubit1();
        let result1 = q1.apply(&h1).unwrap().apply(&h1).unwrap();
        assert!(c64_relative_eq(&result1.v[0], &q1.v[0]));
        assert!(c64_relative_eq(&result1.v[1], &q1.v[1]));

        let h2 = hadamard(2);
        let reg = Register::new(2).unwrap();
        let result2 = reg.apply(&h2).unwrap();
        let half = Complex64 { re: 0.5, im: 0.0 };
        for i in 0..result2.vector.nrows() {
            assert!(c64_relative_eq(&result2.vector[i], &half));
        }
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

    #[test]
    fn test_kronecker_product() {
        let ha = hadamard(1);
        let hb = hadamard(1);

        let h2 = kronecker_product(&ha.matrix, &hb.matrix);

        let half = Complex { re: 0.5, im: 0.0 };

        // note that this appears transposed because matrices are stored in column-major order
        // TODO: the hadamard matrix is symmetric and so doesn't make a good test case
        let expected = [[half,  half,  half,  half],
                        [half, -half,  half, -half],
                        [half,  half, -half, -half],
                        [half, -half, -half,  half]];

        for r in 0..h2.nrows() {
            for c in 0..h2.ncols() {
                assert!(c64_relative_eq(&h2[(r, c)], &expected[c][r]));
            }
        }
    }
}
