#![feature(portable_simd)]

pub mod dim2;
pub mod dim4;
pub mod graphic;
pub mod rand;

mod mat_trait;
mod vec_trait;

pub use mat_trait::Matrix;
pub use vec_trait::Vector;
