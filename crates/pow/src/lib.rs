use std::fmt::{Display, Write};

pub fn pow(a: u128, b: u32) -> u128 {
    a.pow(b)
}

pub fn pow_fast(mut a: u128, mut b: u32) -> u128 {
    if a == 0 {
        return if b >= 1 { 0 } else { 1 };
    }
    let mut res = 1;
    while b > 1 {
        println!("r={res} a={a}, b={b}");
        if b % 2 == 1 {
            res *= a
        }
        a *= a;
        b >>= 1;
    }
    if b == 1 {
        a * res
    } else {
        res
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriState {
    Pos,
    Zero,
    Neg,
}

pub struct SignedBinary(Vec<TriState>);

impl From<u128> for SignedBinary {
    fn from(mut val: u128) -> Self {
        let mut v = vec![];
        while val > 0 {
            let s = match val % 4 {
                0b00 | 0b10 => TriState::Zero,
                0b01 => TriState::Pos,
                0b11 => {
                    val += 1;
                    TriState::Neg
                }
                _ => unreachable!(),
            };
            v.push(s);
            val >>= 1;
        }
        Self(v)
    }
}

impl Display for SignedBinary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in self.0.iter().rev() {
            f.write_char(match i {
                TriState::Pos => '1',
                TriState::Zero => '0',
                TriState::Neg => 'T',
            })?
        }
        Ok(())
    }
}

fn pgcd(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_u, mut u) = (1, 0);
    let (mut old_v, mut v) = (0, 1);

    while r != 0 {
        let q = old_r.div_euclid(r);
        (old_r, r) = (r, old_r - q * r);
        (old_u, u) = (u, old_u - q * u);
        (old_v, v) = (v, old_v - q * v);
    }

    (old_r, old_u, old_v)
}

pub fn pow_signedbinary_mod(mut a: u128, b: SignedBinary, p: u128) -> u128 {
    let (r, u, _) = pgcd(a as i128, p as i128);
    assert_eq!(r, 1);

    let mut inv_a = ((u + p as i128) % p as i128) as u128;
    println!("a={a} inv_a={u}");
    assert_eq!((inv_a * a) % p, 1);

    let mut res = 1;

    for d in b.0 {
        match d {
            TriState::Pos => res = (res * a) % p,
            TriState::Zero => (),
            TriState::Neg => res = (res * inv_a) % p,
        };

        inv_a = (inv_a * inv_a) % p;
        a = (a * a) % p;
    }

    res
}

#[cfg(test)]
mod tests {
    use crate::{pow, pow_fast, pow_signedbinary_mod, SignedBinary};

    #[test]
    fn tpow_fast() {
        for a in [0, 5, 4, 8, 10] {
            for b in [0, 5, 9, 32, 10, 9] {
                println!("Test a={a} b={b}");
                assert_eq!(pow(a, b), pow_fast(a, b))
            }
        }
    }

    #[test]
    fn tpow_trinary() {
        let p = 15168469;
        for a in [5, 4, 8, 6] {
            for b in [15u32, 9, 7, 10, 9] {
                let bsig: SignedBinary = (b as u128).into();
                println!("Test a={a} b={b}, b'={bsig}");
                assert_eq!(pow(a, b) % p, pow_signedbinary_mod(a, bsig, p))
            }
        }
    }
}
