use std::simd::f32x2;

use crate::{Matrix, Vector};

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vec2(pub f32x2);

// Stored in row major order for efficient SIMD multiplication
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mat2(pub [f32x2; 2]);

impl Vec2 {
    #[inline]
    pub fn from_components(x: f32, y: f32) -> Self {
        Self::from_array([x, y])
    }

    pub fn x(self) -> f32 {
        self[0]
    }
    pub fn y(self) -> f32 {
        self[1]
    }
}

impl crate::Vector<2> for Vec2 {
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
        Self(f32x2::splat(f))
    }

    fn from_array(coords: [f32; 2]) -> Self {
        Self(f32x2::from_array(coords))
    }

    fn into_array(self) -> [f32; 2] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl Mat2 {
    pub fn from_array(mat: &[f32; 4]) -> Self {
        Self::from_rows(unsafe { std::mem::transmute(mat) })
    }

    pub fn into_array(self) -> [f32; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl crate::Matrix<2> for Mat2 {
    type VecDIM = Vec2;
    fn transpose(self) -> Self {
        let [a, b, c, d] = self.into_array();
        Self::from_array(&[a, c, b, d])
    }

    fn from_rows(rows: &[[f32; 2]; 2]) -> Self {
        Mat2(rows.map(f32x2::from_array))
    }

    fn identity() -> Self {
        Self::from_rows(&[[1.0, 0.0], [0.0, 1.0]])
    }

    fn into_rows(self) -> [[f32; 2]; 2] {
        unsafe { std::mem::transmute(self.0) }
    }

    fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + other.0[i]))
    }

    fn sub(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] - other.0[i]))
    }

    fn neg(self) -> Self {
        Self(self.0.map(|a| -a))
    }

    fn vec_mul(vec: Self::VecDIM, this: Self) -> Self::VecDIM {
        let v = vec.into_array().map(f32x2::splat);
        Vec2(
            std::array::from_fn::<_, 2, _>(|i| v[i] * this.0[i])
                .into_iter()
                .sum(),
        )
    }
}
