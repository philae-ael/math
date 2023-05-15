use crate::{dim2::Vec2, dim4::Vec4};

pub trait Vector<const DIM: usize>: Sized + Copy {
    fn from_array(arr: [f32; DIM]) -> Self;
    fn into_array(self) -> [f32; DIM];

    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn component_mul(self, other: Self) -> Self;
    fn splat(f: f32) -> Self;
    fn length_squared(self) -> f32 {
        self.component_mul(self).into_array().into_iter().sum()
    }
    fn length(self) -> f32 {
        self.length_squared().sqrt()
    }
}

macro_rules! VectorImpls {
    ($dim: expr, $v: ident) => {
        impl std::ops::Add for $v {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Vector::<$dim>::add(self, rhs)
            }
        }

        impl std::ops::Sub for $v {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Vector::<$dim>::sub(self, rhs)
            }
        }

        impl std::ops::Neg for $v {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Vector::<$dim>::neg(self)
            }
        }

        impl std::ops::Index<usize> for $v {
            type Output = f32;

            fn index(&self, index: usize) -> &Self::Output {
                &self.0.as_array()[index]
            }
        }

        impl std::ops::IndexMut<usize> for $v {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.0.as_mut_array()[index]
            }
        }
    };
}

VectorImpls!(2, Vec2);
VectorImpls!(4, Vec4);
