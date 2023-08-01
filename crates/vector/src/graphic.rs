use std::simd::f32x4;

use crate::{dim4::{Mat4, Vec4}, Vector, Matrix};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3(Vec4);

impl Default for Point3 {
    fn default() -> Self {
        Self(Vec4::from_array([0.0, 0.0, 0.0, 1.0]))
    }
}

impl Point3 {
    pub fn translate(&mut self, x: f32, y: f32, z: f32) {
        self.0 = self.0 + Vec4::from_components(x, y, z, 0.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scale3(Vec4);

impl Default for Scale3 {
    fn default() -> Self {
        Self(Vec4::from_array([1.0, 1.0, 1.0, 0.0]))
    }
}

impl std::ops::Mul<Mat4> for Scale3 {
    type Output = Mat4;

    fn mul(self, rhs: Mat4) -> Self::Output {
        let [l0, l1, l2, _] = self.0.into_array().map(f32x4::splat);
        let [r0, r1, r2, r3] = rhs.0;
        Mat4([l0 * r0, l1 * r1, l2 * r2, r3])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion(Vec4);

impl Default for Quaternion {
    fn default() -> Self {
        Self(Vec4::from_array([1.0, 0.0, 0.0, 0.0]))
    }
}

impl Quaternion {
    pub fn as_mat4(&self) -> Mat4 {
        // 12 mut + 11 add
        let [a, b, c, d] = self.0.into_array();

        let s = 2. / (a * a + b * b + c * c + d * d);

        let bs = b * s;
        let cs = c * s;
        let ds = d * s;

        let ab = a * bs;
        let ac = a * cs;
        let ad = a * ds;

        let bb = b * bs;
        let bc = b * cs;
        let bd = b * ds;

        let cc = c * cs;
        let cd = c * ds;
        let dd = d * ds;

        Mat4::from_rows(&[
            [1. - cc - dd, bc - ad, bd + ac, 0.0],
            [bc + ad, 1. - bb - dd, cd - ab, 0.0],
            [bd - ac, cd + ab, 1. - bb - cc, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
    pub fn from_euler(x: f32, y: f32, z: f32) -> Self {
        let (sx, cx) = f32::sin_cos(x / 2.);
        let (sy, cy) = f32::sin_cos(y / 2.);
        let (sz, cz) = f32::sin_cos(z / 2.);

        Self(Vec4::from_array([
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ]))
    }
}

// First Translate then rotate and finally rotate
#[derive(Default)]
pub struct Transform {
    pub position: Point3,
    pub scale: Scale3,
    pub rotation: Quaternion,
}

impl Transform {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn as_mat4(&self) -> Mat4 {
        // | S * R T |
        // |   0   1 |
        let mut mat = self.scale * self.rotation.as_mat4();
        mat[(0, 3)] = self.position.0[0];
        mat[(1, 3)] = self.position.0[1];
        mat[(2, 3)] = self.position.0[2];
        mat
    }
}

pub struct PerspCamera {
    pub aspect_ratio: f32,
    pub fov: f32,
    pub far: f32,
    pub near: f32,
}

impl PerspCamera {
    pub fn new(aspect_ratio: f32, fov: f32, far: f32, near: f32) -> Self {
        Self {
            aspect_ratio,
            fov,
            far,
            near,
        }
    }

    pub fn as_mat4(&self) -> Mat4 {
        let cotan_half_fov = 1. / f32::tan(self.fov / 2.);
        let m = cotan_half_fov / self.aspect_ratio;
        let l = cotan_half_fov;
        let q = -self.far / (self.far - self.near);
        let q2 = q * self.near;
        Mat4::from_rows(&[
            [m, 0., 0., 0.],
            [0., l, 0., 0.],
            [0., 0., q, q2],
            [0., 0., -1., 0.],
        ])
    }
}
