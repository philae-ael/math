use rand::prelude::Distribution;

use crate::{dim2::Vec2, dim4::Vec4};

pub struct Vec2Uniform {
    inner_x: rand::distributions::Uniform<f32>,
    inner_y: rand::distributions::Uniform<f32>,
}

impl rand::distributions::uniform::UniformSampler for Vec2Uniform {
    type X = Vec2;

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec2 {
        Vec2::from_components(self.inner_x.sample(rng), self.inner_y.sample(rng))
    }

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
    {
        let [low_x, low_y] = low.borrow().into_array();
        let [high_x, high_y] = high.borrow().into_array();
        Self {
            inner_x: rand::distributions::Uniform::new(low_x, high_x),
            inner_y: rand::distributions::Uniform::new(low_y, high_y),
        }
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
    {
        let [low_x, low_y] = low.borrow().into_array();
        let [high_x, high_y] = high.borrow().into_array();
        Self {
            inner_x: rand::distributions::Uniform::new_inclusive(low_x, high_x),
            inner_y: rand::distributions::Uniform::new_inclusive(low_y, high_y),
        }
    }
}

impl rand::distributions::uniform::SampleUniform for Vec2 {
    type Sampler = Vec2Uniform;
}

pub struct Vec4Uniform {
    inner_x: rand::distributions::Uniform<f32>,
    inner_y: rand::distributions::Uniform<f32>,
    inner_z: rand::distributions::Uniform<f32>,
    inner_w: rand::distributions::Uniform<f32>,
}

impl rand::distributions::uniform::UniformSampler for Vec4Uniform {
    type X = Vec4;

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec4 {
        Vec4::from_components(
            self.inner_x.sample(rng),
            self.inner_y.sample(rng),
            self.inner_z.sample(rng),
            self.inner_w.sample(rng),
        )
    }

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
    {
        let [low_x, low_y, low_z, low_w] = low.borrow().into_array();
        let [high_x, high_y, high_z, high_w] = high.borrow().into_array();
        Self {
            inner_x: rand::distributions::Uniform::new(low_x, high_x),
            inner_y: rand::distributions::Uniform::new(low_y, high_y),
            inner_z: rand::distributions::Uniform::new(low_z, high_z),
            inner_w: rand::distributions::Uniform::new(low_w, high_w),
        }
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
    {
        let [low_x, low_y, low_z, low_w] = low.borrow().into_array();
        let [high_x, high_y, high_z, high_w] = high.borrow().into_array();
        Self {
            inner_x: rand::distributions::Uniform::new_inclusive(low_x, high_x),
            inner_y: rand::distributions::Uniform::new_inclusive(low_y, high_y),
            inner_z: rand::distributions::Uniform::new_inclusive(low_z, high_z),
            inner_w: rand::distributions::Uniform::new_inclusive(low_w, high_w),
        }
    }
}

impl rand::distributions::uniform::SampleUniform for Vec4 {
    type Sampler = Vec4Uniform;
}
