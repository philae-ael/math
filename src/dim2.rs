use std::simd::f32x2;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vec2(pub f32x2);

// Stored in row major order for efficient SIMD multiplication
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mat2(pub [f32x2; 2]);

impl Vec2 {
    #[inline]
    pub fn from_array(coords: [f32; 2]) -> Self {
        Self(f32x2::from_array(coords))
    }
    #[inline]
    pub fn from_components(x: f32, y: f32) -> Self {
        Self::from_array([x, y])
    }
    #[inline]
    pub fn component_mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    #[inline]
    pub fn splat(f: f32) -> Self {
        Self(f32x2::splat(f))
    }

    #[inline]
    pub fn into_array(self) -> [f32; 2] {
        unsafe { std::mem::transmute(self.0) }
    }

    pub fn length_squared(self) -> f32 {
        let [x, y] = self.component_mul(self).into_array();
        x + y
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }
}

impl Mat2 {
    pub fn transpose(self) -> Self {
        let [a, b, c, d] = self.into_array();
        // HUHU ?
        Self::from_array(&[a, c, b, d])
    }

    #[inline]
    pub fn from_rows(rows: &[[f32; 2]; 2]) -> Self {
        Mat2(rows.map(f32x2::from_array))
    }

    pub fn from_array(mat: &[f32; 4]) -> Self {
        Self::from_rows(unsafe { std::mem::transmute(mat) })
    }

    pub fn identity() -> Self {
        Self::from_rows(&[[1.0, 0.0], [0.0, 1.0]])
    }

    pub fn into_rows(self) -> [[f32; 2]; 2] {
        unsafe { std::mem::transmute(self.0) }
    }

    pub fn into_array(self) -> [f32; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl std::ops::Index<usize> for Vec2 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0.as_array()[index]
    }
}

impl std::ops::IndexMut<usize> for Vec2 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0.as_mut_array()[index]
    }
}

impl std::ops::Index<(usize, usize)> for Mat2 {
    type Output = f32;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.0[row].as_array()[col]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Mat2 {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.0[row].as_mut_array()[col]
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Add for Mat2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let [l0, l1] = self.0;
        let [r0, r1] = rhs.0;
        Self([l0 + r0, l1 + r1])
    }
}

impl std::ops::Mul<Mat2> for Vec2 {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: Mat2) -> Self::Output {
        let [l0, l1] = self.0.as_array().map(f32x2::splat);
        let [r0, r1] = rhs.0;

        Vec2(r0 * l0 + r1 * l1)
    }
}

impl std::ops::Mul<Vec2> for Mat2 {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: Vec2) -> Self::Output {
        rhs * self.transpose()
    }
}

impl std::ops::Mul for Mat2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.map(|x| (Vec2(x) * rhs).0))
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec2::splat(rhs).component_mul(self)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec2(self.0 - rhs.0)
    }
}

impl std::ops::Neg for Vec2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vec2(-self.0)
    }
}

impl std::ops::Mul<f32> for Mat2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let v = f32x2::splat(rhs);
        Self(self.0.map(|x| v * x))
    }
}
