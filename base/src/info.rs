use std::{borrow::Cow, fmt::Display};

use crate::opinion::{MyFloat, Phi, Psi, B, H, O};
use subjective_logic::{
    mul::labeled::{OpinionD1, SimplexD1},
    multi_array::labeled::MArrD1,
};

#[derive(Debug, Clone)]
pub enum InfoContent<'a, V: Clone> {
    Misinfo {
        op: Cow<'a, OpinionD1<Psi, V>>,
    },
    Correction {
        op: Cow<'a, OpinionD1<Psi, V>>,
        misinfo: Cow<'a, OpinionD1<Psi, V>>,
    },
    Observation {
        op: Cow<'a, OpinionD1<O, V>>,
    },
    Inhibition {
        op1: Cow<'a, OpinionD1<Phi, V>>,
        op2: Cow<'a, MArrD1<Psi, SimplexD1<H, V>>>,
        op3: Cow<'a, MArrD1<B, SimplexD1<H, V>>>,
    },
}

impl<'a, V: MyFloat> From<&InfoContent<'a, V>> for InfoLabel {
    fn from(value: &InfoContent<'a, V>) -> Self {
        match value {
            InfoContent::Misinfo { .. } => InfoLabel::Misinfo,
            InfoContent::Correction { .. } => InfoLabel::Corrective,
            InfoContent::Observation { .. } => InfoLabel::Observed,
            InfoContent::Inhibition { .. } => InfoLabel::Inhibitive,
        }
    }
}

#[derive(Debug)]
pub struct Info<'a, V: Clone> {
    pub idx: usize,
    num_shared: usize,
    num_viewed: usize,
    label: InfoLabel,
    p: InfoContent<'a, V>,
}

impl<'a, V: MyFloat> Info<'a, V> {
    pub fn new(idx: usize, p: InfoContent<'a, V>) -> Self {
        Self {
            idx,
            label: (&p).into(),
            num_shared: 0,
            num_viewed: 0,
            p,
        }
    }

    pub fn content(&self) -> &InfoContent<'a, V> {
        &self.p
    }

    pub fn label(&self) -> &InfoLabel {
        &self.label
    }

    #[inline]
    pub fn viewed(&mut self) {
        self.num_viewed += 1;
    }

    #[inline]
    pub fn shared(&mut self) {
        self.num_shared += 1;
    }

    #[inline]
    pub fn num_shared(&self) -> usize {
        self.num_shared
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Copy)]
pub enum InfoLabel {
    Misinfo,
    Corrective,
    Observed,
    Inhibitive,
}

impl From<&InfoLabel> for u8 {
    fn from(value: &InfoLabel) -> Self {
        match value {
            InfoLabel::Misinfo => 0,
            InfoLabel::Corrective => 1,
            InfoLabel::Observed => 2,
            InfoLabel::Inhibitive => 3,
        }
    }
}

impl Display for InfoLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InfoLabel::Misinfo => write!(f, "misinfo"),
            InfoLabel::Corrective => write!(f, "corrective"),
            InfoLabel::Observed => write!(f, "observed"),
            InfoLabel::Inhibitive => write!(f, "inhivitive"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InfoLabel;

    #[test]
    fn test_info_type() {
        println!("{}", InfoLabel::Misinfo);
    }
}
