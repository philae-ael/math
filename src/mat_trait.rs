use crate::{dim2::Mat2, dim4::Mat4, Vector};

pub trait Matrix<const DIM: usize>: Sized + Copy {
    type VecDIM: Vector<DIM>;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn neg(self) -> Self;

    fn transpose(self) -> Self;
    fn from_rows(rows: &[[f32; DIM]; DIM]) -> Self;
    fn identity() -> Self;
    fn into_rows(self) -> [[f32; DIM]; DIM];
    fn vec_mul(vec: Self::VecDIM, this: Self) -> Self::VecDIM;
    fn mul_vec(this: Self, vec: Self::VecDIM) -> Self::VecDIM {
        Self::vec_mul(vec, this.transpose())
    }
    fn mul(self, other: Self) -> Self {
        Self::from_rows(
            &self
                .into_rows()
                .map(Self::VecDIM::from_array)
                .map(|x| Self::vec_mul(x, other).into_array()),
        )
    }
}

macro_rules! MatrixImpls {
    ($dim: expr, $m: ident) => {
        impl std::ops::Add for $m {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Matrix::<$dim>::add(self, rhs)
            }
        }

        impl std::ops::Sub for $m {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Matrix::<$dim>::sub(self, rhs)
            }
        }

        impl std::ops::Neg for $m {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Matrix::<$dim>::neg(self)
            }
        }

        impl std::ops::Index<(usize, usize)> for $m {
            type Output = f32;

            fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
                &self.0[row].as_array()[col]
            }
        }

        impl std::ops::IndexMut<(usize, usize)> for $m {
            fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
                &mut self.0[row].as_mut_array()[col]
            }
        }

        impl std::ops::Mul<<Self as Matrix<$dim>>::VecDIM> for $m {
            type Output = <Self as Matrix<$dim>>::VecDIM;

            fn mul(self, rhs: Self::Output) -> Self::Output {
                Matrix::<$dim>::mul_vec(self, rhs)
            }
        }

        impl std::ops::Mul<$m> for <$m as Matrix<$dim>>::VecDIM {
            type Output = Self;

            fn mul(self, rhs: $m) -> Self::Output {
                Matrix::<$dim>::vec_mul(self, rhs)
            }
        }

        impl std::ops::Mul for $m {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                Matrix::<$dim>::mul(self, rhs)
            }
        }
    };
}

MatrixImpls!(2, Mat2);
MatrixImpls!(4, Mat4);
