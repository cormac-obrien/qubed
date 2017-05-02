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

extern crate num;
extern crate qubed;

use num::{One, Zero};
use qubed::Qubit;
use qubed::complex::Complex64;

fn main() {
    let ha = qubed::hadamard(1);
    let hb = qubed::hadamard(1);

    let h2 = qubed::kronecker_product(&ha.matrix, &hb.matrix);

    let simulations: usize = 10000000;

    println!("Running {} simulations", simulations);

    let h = qubed::hadamard(1);
    let qubit = Qubit::new(Complex64::zero(), Complex64::one()).unwrap();
    let result = qubit.apply(&h).unwrap();

    let mut ones: usize = 0;
    for i in 0..simulations + 1 {
        if i % 1000 == 0 {
            print!("progress: {:8}/{:8} ({:.0}%)|| distribution: {:.2}\r",
                   i,
                   simulations,
                   (i as f64 / simulations as f64) * 100.0,
                   (ones as f64 / i as f64) * 100.0);
        }

        if i >= simulations {
            break;
        }

        if result.collapse() {
            ones += 1;
        }
    }

    println!("");
}
