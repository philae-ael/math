use vector::{dim2::Vec2, Vector};

#[derive(Debug, Clone, Copy)]
pub struct Extent {
    min: Vec2,
    max: Vec2,
}

impl Extent {
    fn contains(&self, pos: Vec2) -> bool {
        let [min_x, min_y] = self.min.into_array();
        let [max_x, max_y] = self.max.into_array();
        let [x, y] = pos.into_array();

        min_x <= x && min_y <= y && x <= max_x && y <= max_y
    }
}

pub enum QuadtreeNode<T, const MAX_DATA_PER_NODE: usize = 8> {
    Node {
        data: Vec<(Vec2, T)>,
        extent: Extent,
    },
    Parent {
        childs: Box<[QuadtreeNode<T, MAX_DATA_PER_NODE>; 4]>,
        extent: Extent,
    },
}

impl<T, const MAX_DATA_PER_NODE: usize> QuadtreeNode<T, MAX_DATA_PER_NODE> {
    pub fn new(extent: [Vec2; 2]) -> Self {
        Self::Node {
            data: Vec::with_capacity(MAX_DATA_PER_NODE),
            extent: Extent {
                min: extent[0],
                max: extent[1],
            },
        }
    }

    pub fn insert(&mut self, pos: Vec2, new_data: T) {
        assert!(self.contains(pos));
        match self {
            QuadtreeNode::Node { data, .. } if data.len() >= MAX_DATA_PER_NODE => {
                self.split();
                self.insert(pos, new_data);
            }
            QuadtreeNode::Node { data, .. } => {
                data.push((pos, new_data));
            }

            QuadtreeNode::Parent { childs, .. } => {
                if let Some(child) = childs.iter_mut().find(|child| child.contains(pos)) {
                    child.insert(pos, new_data)
                }
            }
        }
    }

    fn contains(&self, pos: Vec2) -> bool {
        match self {
            QuadtreeNode::Node { extent, .. } => extent.contains(pos),
            QuadtreeNode::Parent { extent, .. } => extent.contains(pos),
        }
    }

    fn split(&mut self) {
        if let QuadtreeNode::Node { data, extent } = self {
            let mid = 0.5 * (extent.min + extent.max);

            let mut out = QuadtreeNode::Parent {
                childs: Box::new([
                    Self::new([
                        Vec2::from_components(extent.min.x(), mid.y()),
                        Vec2::from_components(mid.x(), extent.max.y()),
                    ]),
                    Self::new([mid, extent.max]),
                    Self::new([extent.min, mid]),
                    Self::new([
                        Vec2::from_components(mid.x(), extent.min.y()),
                        Vec2::from_components(extent.max.x(), mid.y()),
                    ]),
                ]),
                extent: *extent,
            };

            for (pos, data) in data.drain(..) {
                out.insert(pos, data);
            }
            *self = out;
        }
    }
}
