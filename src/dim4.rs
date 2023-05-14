use std::simd::f32x4;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vec4(pub f32x4);

// Stored in row major order for efficient SIMD multiplication
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mat4(pub [f32x4; 4]);

impl Vec4 {
    #[inline]
    pub fn from_array(coords: [f32; 4]) -> Self {
        Self(f32x4::from_array(coords))
    }
    #[inline]
    pub fn from_components(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self::from_array([x, y, z, w])
    }
    #[inline]
    pub fn component_mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    #[inline]
    pub fn splat(f: f32) -> Self {
        Self(f32x4::splat(f))
    }

    #[inline]
    pub fn into_array(self) -> [f32; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl Mat4 {
    #[cfg(target_feature = "sse")]
    pub fn transpose(self) -> Self {
        let [mut a, mut b, mut c, mut d] = self.0.map(core::arch::x86_64::__m128::from);
        unsafe { core::arch::x86_64::_MM_TRANSPOSE4_PS(&mut a, &mut b, &mut c, &mut d) };
        Self([a, b, c, d].map(f32x4::from))
    }

    #[inline]
    pub fn from_rows(rows: &[[f32; 4]; 4]) -> Self {
        Mat4(rows.map(f32x4::from_array))
    }

    pub fn from_array(mat: &[f32; 16]) -> Self {
        Self::from_rows(unsafe { std::mem::transmute(mat) })
    }

    pub fn identity() -> Self {
        Self::from_rows(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn into_rows(self) -> [[f32; 4]; 4] {
        unsafe { std::mem::transmute(self.0) }
    }

    pub fn into_array(self) -> [f32; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl std::ops::Index<usize> for Vec4 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0.as_array()[index]
    }
}

impl std::ops::IndexMut<usize> for Vec4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0.as_mut_array()[index]
    }
}

impl std::ops::Index<(usize, usize)> for Mat4 {
    type Output = f32;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.0[row].as_array()[col]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Mat4 {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.0[row].as_mut_array()[col]
    }
}

impl std::ops::Add for Vec4 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Add for Mat4 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let [l0, l1, l2, l3] = self.0;
        let [r0, r1, r2, r3] = rhs.0;
        Self([l0 + r0, l1 + r1, l2 + r2, l3 + r3])
    }
}

impl std::ops::Mul<Mat4> for Vec4 {
    type Output = Vec4;

    #[inline]
    fn mul(self, rhs: Mat4) -> Self::Output {
        let [l0, l1, l2, l3] = self.0.as_array().map(f32x4::splat);
        let [r0, r1, r2, r3] = rhs.0;

        Vec4(r0 * l0 + r1 * l1 + r2 * l2 + r3 * l3)
    }
}

impl std::ops::Mul<Vec4> for Mat4 {
    type Output = Vec4;

    #[inline]
    fn mul(self, rhs: Vec4) -> Self::Output {
        rhs * self.transpose()
    }
}

impl std::ops::Mul for Mat4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.map(|x| (Vec4(x) * rhs).0))
    }
}

impl std::ops::Mul<f32> for Vec4 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec4::splat(rhs).component_mul(self)
    }
}

impl std::ops::Mul<f32> for Mat4 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let v = f32x4::splat(rhs);
        Self(self.0.map(|x| v * x))
    }
}
