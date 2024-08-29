use std::{
    borrow::Cow,
    ffi::OsStr,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::anyhow;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct MyPath(PathBuf);

impl From<String> for MyPath {
    #[inline]
    fn from(value: String) -> Self {
        Self(value.into())
    }
}

impl FromStr for MyPath {
    type Err = <PathBuf as FromStr>::Err;
    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PathBuf::from_str(s).map(Self)
    }
}

impl<T: ?Sized + AsRef<OsStr>> From<&T> for MyPath {
    #[inline]
    fn from(value: &T) -> Self {
        Self(value.into())
    }
}

impl MyPath {
    pub fn join_path<P: AsRef<Path>>(&mut self, root: P) {
        if !self.0.is_absolute() {
            self.0 = root.as_ref().join(&self.0);
        }
    }

    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        self.0.to_string_lossy()
    }

    pub fn verify(&self) -> anyhow::Result<&Path> {
        if self.0.try_exists()? {
            Ok(self.0.as_path())
        } else {
            Err(anyhow!("Path {} does not exist.", self.to_string_lossy()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MyPath;

    #[test]
    fn test_my_path() {
        let p: MyPath = "./hoge/fuga".into();
        let q = p.verify();
        println!("{q:?}");
        assert!(q.is_err());
        let p: MyPath = "./src/".into();
        assert!(p.verify().is_ok());
        let p: MyPath = "./src/io.rs".into();
        assert!(p.verify().is_ok());
    }
}
