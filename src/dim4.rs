use std::simd::f32x4;

use crate::{Matrix, Vector};

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vec4(pub f32x4);

// Stored in row major order for efficient SIMD multiplication
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mat4(pub [f32x4; 4]);

impl Vec4 {
    #[inline]
    pub fn from_components(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self::from_array([x, y, z, w])
    }
}

impl Vector<4> for Vec4 {
    fn from_array(coords: [f32; 4]) -> Self {
        Self(f32x4::from_array(coords))
    }

    fn into_array(self) -> [f32; 4] {
        unsafe { std::mem::transmute(self.0) }
    }

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    fn neg(self) -> Self {
        Self(-self.0)
    }

    fn component_mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    fn splat(f: f32) -> Self {
        Self(f32x4::splat(f))
    }
}

impl Mat4 {
    pub fn from_array(mat: &[f32; 16]) -> Self {
        Self::from_rows(unsafe { std::mem::transmute(mat) })
    }

    pub fn into_array(self) -> [f32; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl Matrix<4> for Mat4 {
    type VecDIM = Vec4;

    fn add(self, other: Self) -> Self {
        Self(self.0.zip(other.0).map(|(a, b)| a + b))
    }

    fn sub(self, other: Self) -> Self {
        Self(self.0.zip(other.0).map(|(a, b)| a - b))
    }

    fn neg(self) -> Self {
        Self(self.0.map(|a| -a))
    }

    fn transpose(self) -> Self {
        let [mut a, mut b, mut c, mut d] = self.0.map(core::arch::x86_64::__m128::from);
        unsafe { core::arch::x86_64::_MM_TRANSPOSE4_PS(&mut a, &mut b, &mut c, &mut d) };
        Self([a, b, c, d].map(f32x4::from))
    }

    fn from_rows(rows: &[[f32; 4]; 4]) -> Self {
        Mat4(rows.map(f32x4::from_array))
    }

    fn identity() -> Self {
        Self::from_rows(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    fn into_rows(self) -> [[f32; 4]; 4] {
        unsafe { std::mem::transmute(self.0) }
    }

    fn vec_mul(vec: Self::VecDIM, this: Self) -> Self::VecDIM {
        Vec4(
            vec.into_array()
                .map(f32x4::splat)
                .zip(this.0)
                .map(|(a, b)| a * b)
                .into_iter()
                .sum(),
        )
    }
}
